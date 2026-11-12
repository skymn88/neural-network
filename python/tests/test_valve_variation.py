"""Opt-in stress input must preserve legacy arrays and isolate hidden truth."""

import unittest
import numpy as np
from actuator.data import pulse_flights, fingerprint


class ValveVariationTests(unittest.TestCase):
    def test_default_zero_and_determinism(self):
        kw = dict(seed=55, durations=(900, 950), profile="flightLike")
        base, _ = pulse_flights(**kw)
        for v in ({}, dict(gain_sd=0, bias_sd=0, drift_amp=0)):
            same, _ = pulse_flights(**kw, valve_variation=v)
            self.assertEqual(fingerprint(base), fingerprint(same))
        setting = dict(gain_sd=0.03, bias_sd=0.025, drift_amp=0.02)
        varied, groups = pulse_flights(**kw, valve_variation=setting)
        again, _ = pulse_flights(**kw, valve_variation=setting)
        self.assertEqual(fingerprint(varied), fingerprint(again))
        self.assertNotEqual(fingerprint(base), fingerprint(varied))
        for a, b in zip(base, varied):
            np.testing.assert_allclose(
                a.truth["ddv_clean"], b.truth["ddv_clean"], atol=5e-4
            )
            self.assertGreater(np.ptp(a.values[:, 0] - b.values[:, 0]), 0.01)
            self.assertEqual(b.truth["valve_bias"].shape, b.time.shape)
            x, rate = b.truth["ram_clean"], b.truth["rate_requested"]
            np.testing.assert_allclose(
                x[1:], np.clip(x[:-1] + rate[1:] / 64, -100, 100), atol=1e-12
            )
        self.assertEqual(set(groups), {"train", "validation", "test"})

    def test_invalid_settings_rejected(self):
        for v in (
            {"gain_sd": -1},
            {"bias_sd": float("nan")},
            {"typo": 0.1},
            {"drift_amp": float("inf")},
        ):
            with self.assertRaises(ValueError):
                pulse_flights(profile="flightLike", durations=(2,), valve_variation=v)
        with self.assertRaises(ValueError):
            pulse_flights(valve_variation={"bias_sd": 0.1})

    def test_predicted_gate_uses_training_prediction_scale(self):
        from unittest.mock import patch
        from actuator.null_rate import fit_null_rate

        flights, _ = pulse_flights(seed=55, durations=(900,), profile="flightLike")
        z = flights[0].values[:, 1]
        with patch("actuator.null_rate.predict_ddv", return_value=z + 0.04):
            old = fit_null_rate(flights, {}, {}, knots=9)
            new = fit_null_rate(flights, {}, {}, knots=9, gate_source="predicted")
        np.testing.assert_allclose(
            np.asarray(new["ram"]["observer"]["gate"]) - old["ram"]["observer"]["gate"],
            0.04,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            new["ram"]["observer"]["center"] - old["ram"]["observer"]["center"], 0.04
        )
        with self.assertRaises(ValueError):
            fit_null_rate(flights, {}, {}, gate_source="truth")

    def test_quiet_window_is_opt_in_and_causal(self):
        from unittest.mock import patch
        from actuator.null_rate import fit_null_rate
        from actuator.inference import neutral_features

        flights, _ = pulse_flights(seed=55, durations=(900,), profile="flightLike")
        z = flights[0].values[:, 1]
        with patch("actuator.null_rate.predict_ddv", return_value=z):
            m = fit_null_rate(
                flights,
                {},
                {},
                knots=9,
                observer="stationary",
                adaptive=True,
                quiet_seconds=2,
                quiet_quantile=0.5,
            )
        spec = m["ram"]["observer"]
        self.assertEqual(spec["quiet_window"], 128)
        full, _ = neutral_features(z, spec)
        prefix, _ = neutral_features(z[:700], spec)
        np.testing.assert_array_equal(prefix, full[:700])
        for kw in (
            {"quiet_seconds": 0},
            {"quiet_quantile": 1},
            {"quiet_seconds": float("nan")},
        ):
            with self.assertRaises(ValueError):
                fit_null_rate(flights, {}, {}, **kw)
