import copy
import unittest
import numpy as np
from actuator.data import Flight, pulse_flights
from actuator.inference import rollout
from actuator.sequence_evaluate import fast_rollout, fast_score
from actuator.null_rate import fit_null_rate


class TrainSearchTests(unittest.TestCase):
    def test_fast_recurrent_rollout_carries_state_across_chunks(self):
        rng = np.random.default_rng(91)
        ref = dict(
            kind="null_rate",
            bounds=[-3, 3],
            knots=[-0.1, 0, 0.1],
            values=[-1, 0, 1],
            delay=1,
            zero_band=[-0.001, 0.001],
            observer=dict(
                center=0, gate=[-0.01, 0.01], tau=2, max_step=0.01, adaptive=False
            ),
        )
        base = dict(
            ddv=dict(scale=1, lags=0, basis="legacy", weights=[0, 1, 0, 0, 0]),
            angle=dict(scale=1, weights=[0, 1]),
            ram=ref,
        )
        t = np.arange(8400) / 64
        current = 0.03 * np.sin(t)
        np.testing.assert_array_equal(
            fast_rollout(base, current, 0)[1], rollout(base, current, 0)[1]
        )
        for family, gates in [("gru", 3), ("lstm", 4), ("elman", 1)]:
            spec = dict(
                family=family,
                hidden=3,
                mean=[0] * 5,
                scale=[1] * 5,
                head=[0.1, 0.2, 0.3],
                bias=0.01,
                wi=rng.normal(0, 0.1, (gates * 3, 5)).tolist(),
                wh=rng.normal(0, 0.1, (gates * 3, 3)).tolist(),
                bi=[0] * (gates * 3),
                bh=[0] * (gates * 3),
                rate_scale=1,
                rate_limit=3,
                gate=True,
                residual=True,
            )
            m = copy.deepcopy(base)
            m["ram"] = dict(
                kind="sequence_rate", reference=ref, bounds=[-3, 3], sequence=spec
            )
            slow = rollout(m, current, 0.1)
            fast = fast_rollout(m, current, 0.1)
            np.testing.assert_allclose(fast, slow, atol=1e-10)
            truth = slow[1] + 0.01 * np.sin(t)
            f = Flight("x", t, np.c_[current, current, truth, truth])
            fit, rmse = fast_score(m, [f])
            self.assertTrue(np.isfinite(fit))
            self.assertAlmostEqual(
                rmse, np.sqrt(np.mean((slow[1][32:] - truth[32:]) ** 2)), places=10
            )

    def test_gate_expansion_preserves_midpoint(self):
        from unittest.mock import patch

        flights, _ = pulse_flights(seed=55, durations=(900,), profile="flightLike")
        with patch(
            "actuator.null_rate.predict_ddv", return_value=flights[0].values[:, 1]
        ):
            a = fit_null_rate(flights, {}, {}, knots=9)
            b = fit_null_rate(flights, {}, {}, knots=9, gate_expansion=2)
        ga = a["ram"]["observer"]["gate"]
        gb = b["ram"]["observer"]["gate"]
        self.assertAlmostEqual(np.mean(ga), np.mean(gb))
        self.assertAlmostEqual(np.ptp(gb), 2 * np.ptp(ga))
        for value in (0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_null_rate(flights, {}, {}, gate_expansion=value)

    def test_selection_basis_changes_ranking_without_using_test(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from actuator import experiment

        t = np.arange(512) / 64
        u = np.sin(t)
        x = np.cumsum(u) / 64
        x -= x[0]
        train = Flight("train", t, np.c_[u, u, x, x])
        val = Flight("val", t, np.c_[u, u, -x, -x])
        groups = {"train": [train], "validation": [val], "test": []}
        ddv = dict(scale=1, lags=0, basis="legacy", weights=[0, 1, 0, 0, 0])
        angle = dict(scale=1, weights=[0, 1])
        models = {
            name: dict(
                ddv=ddv,
                angle=angle,
                ram=dict(kind="oe", scale=1, lags=0, weights=[0, sign, 0, 0], beta=0),
            )
            for name, sign in [("train_leader", 1), ("validation_leader", -1)]
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(experiment, "ROOT", Path(tmp)),
            patch.object(experiment, "fit_ddv", return_value=(ddv, [])),
            patch.object(experiment, "fit_angle", return_value=angle),
            patch.object(experiment, "fit_neural", return_value=(models, [])),
        ):
            normal, _, _ = experiment.run(
                [train, val],
                groups,
                source="fixture",
                methods=["neural"],
                log=lambda _: None,
            )
            targeted, _, _ = experiment.run(
                [train, val],
                groups,
                source="fixture",
                methods=["neural"],
                selectionBasis="train",
                log=lambda _: None,
            )
        self.assertEqual(normal["selected"], "validation_leader")
        self.assertEqual(targeted["selected"], "train_leader")
        self.assertIn("TRAIN-selected", targeted["notice"])
