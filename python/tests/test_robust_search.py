import tempfile
import unittest
from pathlib import Path
import numpy as np
from actuator.data import Flight
from actuator.inference import rollout
from actuator.robust_search import (
    search,
    selection_folds,
    flight_stats,
    rank_key,
    fit_candidate,
)
from actuator.seed_ensemble import make_ensemble


def fixture():
    t = np.arange(512) / 64
    flights = []
    for i in range(6):
        u = 0.1 * np.sin(t * (1 + i / 10))
        x = np.r_[0, np.cumsum(u[1:]) / 64]
        flights.append(Flight(str(i), t, np.column_stack([u, u, x, -x])))
    return flights, dict(train=flights[:3], validation=flights[3:4], test=flights[4:])


def model():
    return dict(
        ddv=dict(lags=0, scale=1, basis="legacy", weights=[0, 1, 0, 0, 0]),
        angle=dict(scale=1, weights=[0, -1]),
        ram=dict(kind="oe", scale=1, lags=0, beta=0, weights=[0, 1, 0, 0]),
    )


class RobustTests(unittest.TestCase):
    def test_lofo_and_checkpoint_no_test(self):
        flights, groups = fixture()
        seen = []

        def trainer(train, cfg, seed, check, cache):
            check()
            seen.append(tuple(f.name for f in train))
            return model()

        with tempfile.TemporaryDirectory() as tmp:
            r, models, out = search(
                flights,
                groups,
                max_candidates=1,
                max_minutes=1,
                trainer=trainer,
                out=Path(tmp),
                log=lambda _: None,
            )
            self.assertTrue(r["selected"])
            self.assertEqual(r["config"]["selectionBasis"], "leave-one-flight-out")
            self.assertEqual(len(seen), 12)
            self.assertTrue(all(set(s) <= set("012") for s in seen))
            self.assertTrue(all(row["flight"] in "012" for row in r["per_flight"]))
            self.assertTrue((out / "search.partial.json").is_file())
            self.assertEqual(len(models[r["selected"]]["ram"]["members"]), 3)
            import openpyxl

            book = openpyxl.load_workbook(out / "comparison.xlsx")
            self.assertIn("CandidateFlights", book.sheetnames)
            book.close()

    def test_rejection_rank_and_rate_average(self):
        self.assertEqual(
            flight_stats([dict(stage="RAM_Cascade", flight="bad", fit=-2)])[
                "rejected_flights"
            ],
            ["bad"],
        )
        self.assertLess(
            rank_key(dict(eligible=True, min_fit=20, mean_fit=95)),
            rank_key(dict(eligible=True, min_fit=21, mean_fit=70)),
        )
        a = model()
        b = model()
        b["ram"]["weights"][1] = 3
        ensemble = make_ensemble([a, b], [1, 2])
        u = np.ones(100) * 0.1
        expected = np.arange(100) * 0.2 / 64
        np.testing.assert_allclose(rollout(ensemble, u, 0)[1], expected)
        b["ddv"]["weights"][1] = 2
        ensemble = make_ensemble([a, b], [1, 2])
        np.testing.assert_allclose(
            rollout(ensemble, u, 0)[1], np.arange(100) * 0.35 / 64
        )
        np.testing.assert_allclose(rollout(ensemble, u, 0, ddv_override=u)[1], expected)
        with self.assertRaises(ValueError):
            make_ensemble([a, b], [1, 1])

    def test_cancel_partial_no_selection(self):
        flights, groups = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            r, _, out = search(
                flights, groups, cancel=lambda: True, out=Path(tmp), log=lambda _: None
            )
            self.assertIsNone(r["selected"])
            self.assertTrue(r["stopped"])
            self.assertTrue((out / "search.partial.json").exists())
        groups["train"] = groups["train"][:1]
        with self.assertRaises(ValueError):
            selection_folds(groups)

    def test_actual_family_smoke(self):
        _, groups = fixture()
        cache = {}
        for family in ("null_rate", "gru_rate", "lstm_rate", "oe", "neural", "stable"):
            with self.subTest(family=family):
                cfg = dict(
                    family=family,
                    hidden=4,
                    horizon=96,
                    steps=2,
                    clip=family in ("gru_rate", "lstm_rate"),
                )
                fitted = fit_candidate(groups["train"], cfg, 42, lambda: None, cache)
                predicted = rollout(fitted, groups["validation"][0].values[:, 0], 0)[1]
                self.assertEqual(predicted.shape, (512,))
                self.assertTrue(np.isfinite(predicted).all())
