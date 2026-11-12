import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from actuator.data import Flight, ROOT
from actuator import experiment


class OfflineAndObserverTests(unittest.TestCase):
    def test_launcher_never_downloads(self):
        start = (ROOT / "python/start.bat").read_text().lower()
        setup = (ROOT / "python/setup_offline.bat").read_text().lower()
        self.assertNotIn("pip install", start)
        self.assertNotIn("https://", start)
        self.assertIn("--no-index", setup)
        self.assertIn("--isolated", setup)
        self.assertIn("--find-links", setup)

    def test_training_forwards_observer_option(self):
        t = np.arange(512) / 64
        flights = [
            Flight(str(i), t, np.c_[np.sin(t), np.sin(t), t + i, 0.3 * (t + i)])
            for i in range(4)
        ]
        groups = {"train": flights[:2], "validation": flights[2:3], "test": flights[3:]}

        def fake_fit(train, ddv, angle, **kwargs):
            return {
                "ddv": ddv,
                "angle": angle,
                "ram": {
                    "kind": "oe",
                    "scale": 1.0,
                    "lags": 0,
                    "weights": [0.0, 1.0, 0.0, 0.0],
                    "beta": 0.0,
                },
            }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(experiment, "ROOT", Path(tmp)),
            patch("actuator.null_rate.fit_null_rate", side_effect=fake_fit) as fit,
        ):
            r, _, _ = experiment.run(
                flights,
                groups,
                source="fixture",
                methods=["null_rate"],
                nullObserver="envelope",
                nullGateExpansion=1.5,
                log=lambda _: None,
            )
        self.assertEqual(fit.call_args.kwargs["observer"], "envelope")
        self.assertEqual(fit.call_args.kwargs["gate_expansion"], 1.5)
        self.assertEqual(r["config"]["nullGateExpansion"], 1.5)
        self.assertEqual(r["config"]["nullObserver"], "envelope")


if __name__ == "__main__":
    unittest.main()
