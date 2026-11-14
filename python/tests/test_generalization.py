import copy, json, unittest
import numpy as np
from actuator.data import Flight, ROOT, fingerprint
from actuator.generalization import (
    generalization_data,
    training_speed_limit,
    clipped_reference,
    make_rate_ensemble,
)
from actuator.sequence_inference import sequence_features, combine_rate
from actuator.sequence_evaluate import fast_rollout
from actuator.inference import rollout
from run_generalization import validation_rank


class GeneralizationTests(unittest.TestCase):
    def fixture(self):
        ref = dict(
            kind="null_rate",
            bounds=[-1, 1],
            knots=[-1, 1],
            values=[-5, 5],
            delay=0,
            zero_band=[-0.001, 0.001],
            observer=dict(
                center=0, gate=[-0.01, 0.01], tau=2, max_step=0.01, adaptive=False
            ),
        )
        spec = dict(
            family="tdnn",
            mean=[0] * 5,
            scale=[1] * 5,
            offsets=[0],
            w=[[0] * 5],
            b=[0],
            head=[0],
            bias=2,
            rate_scale=1,
            rate_limit=10,
            residual=False,
            gate=False,
        )
        return dict(
            ddv=dict(scale=1, lags=0, basis="legacy", weights=[0, 1, 0, 0, 0]),
            angle=dict(scale=1, weights=[0, 1]),
            ram=dict(
                kind="sequence_rate", bounds=[-1, 1], reference=ref, sequence=spec
            ),
        )

    def test_rate_average_precedes_position_clamp(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b["ram"]["sequence"]["bias"] = -1
        e = make_rate_ensemble([a, b], [42, 43])
        u = np.zeros(130)
        y = rollout(e, u, 0.9)[1]
        wrong = (rollout(a, u, 0.9)[1] + rollout(b, u, 0.9)[1]) / 2
        self.assertAlmostEqual(y[-1], 1)
        self.assertGreater(abs(y[-1] - wrong[-1]), 0.9)
        np.testing.assert_allclose(
            fast_rollout(e, u, 0.9), rollout(e, u, 0.9), atol=1e-12
        )
        np.testing.assert_allclose(rollout(e, u[:50], 0.9)[1], y[:50])
        with self.assertRaises(ValueError):
            make_rate_ensemble([a, b], [42, 42])
        bad = copy.deepcopy(b)
        bad["ddv"]["scale"] = 2
        with self.assertRaises(ValueError):
            make_rate_ensemble([a, bad], [42, 43])

    def test_q999_map_correction_and_total_caps(self):
        m = self.fixture()
        r = clipped_reference(dict(ram=m["ram"]["reference"]), 0.5)["ram"]
        features, base = sequence_features(np.array([0.0, 1.0, -1.0]), r)
        self.assertLessEqual(abs(base).max(), 0.5)
        spec = dict(
            m["ram"]["sequence"],
            correction_limit=0.5,
            output_rate_limit=0.5,
            residual=True,
        )
        rate = combine_rate(spec, np.array([20.0, -20.0, 20.0]), base)
        self.assertLessEqual(abs(rate).max(), 0.5)
        np.testing.assert_allclose(rate, [0.5, 0, 0])
        self.assertNotIn("map_rate_limit", m["ram"]["reference"])
        t = np.arange(1001) / 64
        v = np.linspace(0, 3, 1000)
        x = np.r_[0, np.cumsum(v) / 64]
        f = Flight("train", t, np.c_[x, x, x, x])
        self.assertAlmostEqual(training_speed_limit([f]), np.quantile(v, 0.999))

    def test_original_split_and_disjoint_frozen_cohorts(self):
        old, split, extra, val = generalization_data()
        prior = json.loads(
            (
                ROOT / "python/audit_results/train_search_20261112_170014/result.json"
            ).read_text()
        )
        self.assertEqual(fingerprint(old), prior["dataset_sha256"])
        self.assertEqual(
            {s: [f.name for f in fs] for s, fs in split.items()}, prior["split"]
        )
        self.assertEqual((len(split["train"]), len(extra), len(val)), (7, 14, 6))
        groups = [{f.name for f in fs} for fs in (old, extra, val)]
        self.assertFalse(
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )

    def test_ranking_ignores_train_and_component(self):
        def rows(a, b):
            return [
                dict(split="validation", stage="RAM_Cascade", fit=a),
                dict(split="validation", stage="RAM_Cascade", fit=b),
                dict(split="train", stage="RAM_Cascade", fit=100),
                dict(split="validation", stage="RAM_Component", fit=100),
            ]

        self.assertGreater(validation_rank(rows(80, 81)), validation_rank(rows(70, 99)))
        self.assertEqual(validation_rank(rows(80, 90)), (80, 85))
