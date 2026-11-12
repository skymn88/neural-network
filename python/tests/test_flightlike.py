"""Small deterministic checks, not extra performance-search iterations."""

import json
import unittest
from unittest.mock import patch
import numpy as np
from actuator.data import Flight, FS, pulse_flights
from actuator.inference import neutral_features, rollout
from actuator.models import fit_ddv, fit_angle
from actuator.null_rate import fit_null_rate
from actuator.augmentation import augment_train, calibrate_train


class FlightLikeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flights, cls.groups = pulse_flights(
            seed=55, durations=(900, 950, 1000, 1050), profile="flightLike"
        )

    def test_position_is_integrated_not_scripted(self):
        f = self.flights[0]
        truth = f.truth
        x = truth["ram_clean"]
        rate = truth["rate_requested"]
        np.testing.assert_allclose(
            x[1:], np.clip(x[:-1] + rate[1:] / FS, -100, 100), atol=1e-12
        )
        self.assertGreater(np.sum(rate == 0), 200 * FS)
        self.assertTrue(np.all(np.diff(x)[rate[1:] == 0] == 0))

    def test_true_null_is_per_flight_and_drifting(self):
        self.assertNotEqual(
            self.flights[0].truth["base_null"], self.flights[1].truth["base_null"]
        )
        self.assertGreater(np.ptp(self.flights[0].truth["null"]), 0.01)
        z = self.flights[0].truth["ddv_clean"]
        null = self.flights[0].truth["null"]
        band = self.flights[0].truth["deadband"]
        delayed = np.roll(z, 2)
        delayed[:2] = z[0]
        self.assertTrue(
            np.all(
                self.flights[0].truth["rate_requested"][abs(delayed - null) <= band]
                == 0
            )
        )

    def test_decreasing_angle_and_units(self):
        f = self.flights[0]
        x = f.truth["ram_clean"]
        expected = 2 - 0.29 * x + 0.0002 * x * x
        self.assertLess(np.sqrt(np.mean((expected - f.values[:, 3]) ** 2)), 0.02)
        self.assertAlmostEqual(2 - 0.29 * (-100) + 0.0002 * 10000, 33)
        self.assertAlmostEqual(2 - 0.29 * 100 + 0.0002 * 10000, -25)

    def test_causal_neutral_observer(self):
        spec = {
            "center": -0.01,
            "gate": [-0.1, 0.1],
            "tau": 2.0,
            "max_step": 0.004,
            "adaptive": True,
        }
        u = np.r_[np.full(50, 0.4), np.full(200, 0.02)]
        a, c = neutral_features(u, spec)
        b, d = neutral_features(np.r_[u, np.ones(100)], spec)
        np.testing.assert_array_equal(a, b[: len(u)])
        np.testing.assert_array_equal(c, d[: len(u)])
        self.assertAlmostEqual(c[-1], 0.02)

    def test_rate_model_json_and_truth_isolation(self):
        train = self.flights[:2]
        val = self.flights[2:3]
        ddv, _ = fit_ddv(train, val, length=4)
        angle = fit_angle(train, val)
        a = fit_null_rate(train, ddv, angle, adaptive=True, knots=21)
        old = [f.truth for f in train]
        try:
            for f in train:
                f.truth = {"null": "must not read simulator truth"}
            b = fit_null_rate(train, ddv, angle, adaptive=True, knots=21)
            self.assertEqual(a, b)
        finally:
            for f, t in zip(train, old):
                f.truth = t
        f = val[0]
        z, x, angle = rollout(json.loads(json.dumps(a)), f.values[:, 0], f.values[0, 2])
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertLessEqual(x[1:].max(), a["ram"]["bounds"][1])

    def test_augmentation_leaves_real_holdouts_untouched(self):
        groups = {
            "train": self.flights[:2],
            "validation": self.flights[2:3],
            "test": self.flights[3:],
        }
        calibration = calibrate_train(groups["train"])
        dummy = Flight("dummy", np.arange(10) / 64, np.zeros((10, 4)))
        with patch(
            "actuator.augmentation.pulse_flights", return_value=([dummy], {})
        ) as generator:
            augmented, added, c = augment_train(groups, 1)
        self.assertEqual(c, calibration)
        self.assertIs(augmented["test"][0], groups["test"][0])
        self.assertIs(augmented["validation"][0], groups["validation"][0])
        self.assertEqual(len(augmented["train"]), 3)
        self.assertEqual(len(groups["train"]), 2)
        self.assertEqual(generator.call_args.kwargs["calibration"], calibration)
        self.assertTrue(added[0].name.startswith("SYNTH_TRAIN_"))

    def test_zero_augmentation_is_noop(self):
        groups = {
            "train": self.flights[:2],
            "validation": self.flights[2:3],
            "test": self.flights[3:],
        }
        copy, added, cal = augment_train(groups, 0)
        self.assertEqual(added, [])
        self.assertIsNone(cal)
        self.assertIs(copy["train"][0], groups["train"][0])

    def test_bad_profiles_and_duration(self):
        with self.assertRaises(ValueError):
            pulse_flights(profile="bad")
        with self.assertRaises(ValueError):
            pulse_flights(profile="flightLike", durations=(100,))


class AugmentationReportTests(unittest.TestCase):
    def test_ab_freezes_selection_before_test(self):
        import copy
        import tempfile
        from pathlib import Path
        from actuator.augmentation import compare_augmentation

        f = Flight("real", np.arange(100) / 64, np.zeros((100, 4)))
        groups = {"train": [f], "validation": [f], "test": [f]}
        events = []
        base = {
            "selected": "M",
            "config": {},
            "timings": {"total": 1},
            "diagnostics": [],
            "search": [],
            "dataset_sha256": "fixture",
            "split": [],
            "versions": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            counter = iter([2.0, 1.0])

            def fake_run(*args, **kw):
                self.assertFalse(kw["include_test"])
                events.append("fit")
                return copy.deepcopy(base), {"M": {"score": next(counter)}}, Path(tmp)

            def fake_evaluate(models, actual, include_test):
                self.assertIs(actual, groups)
                self.assertTrue(include_test)
                events.append("test")
                return [], []

            with (
                patch("actuator.experiment.run", side_effect=fake_run),
                patch(
                    "actuator.augmentation.augment_train",
                    return_value=(groups, [f], {}),
                ),
                patch(
                    "actuator.augmentation.val_rmse",
                    side_effect=lambda m, fs: m["score"],
                ),
                patch("actuator.augmentation.evaluate", side_effect=fake_evaluate),
                patch("actuator.augmentation.write_workbook"),
            ):
                r, _, _ = compare_augmentation(
                    [f], groups, 1, source="user", include_test=True
                )
            self.assertEqual(events, ["fit", "fit", "test"])
            self.assertEqual(r["selected"], "M__SYNTH")
            self.assertEqual(r["diagnostics"][-1]["validation_rmse_delta"], -1.0)


if __name__ == "__main__":
    unittest.main()
