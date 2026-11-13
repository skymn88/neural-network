import copy
import unittest
import numpy as np
from actuator.head_refine import clipped_trajectory, refine_sequence_head
from actuator.inference import rollout
from actuator.data import Flight
from run_balanced_search import meets_goal


class BalancedSearchTests(unittest.TestCase):
    def test_clipped_trajectory_jacobian(self):
        t = np.arange(300) / 64
        design = np.c_[np.sin(3 * t), np.cos(t), np.ones(len(t))]
        weights = np.array([0.7, 0.2, 0.13])
        offset = np.full(len(t), -0.25)
        x, jac = clipped_trajectory(design, offset, weights, 0, [-0.2, 0.3], 0.6)
        for i in range(3):
            pert = weights.copy()
            pert[i] += 1e-7
            y, _ = clipped_trajectory(design, offset, pert, 0, [-0.2, 0.3], 0.6)
            np.testing.assert_allclose((y - x) / 1e-7, jac[:, i], atol=1e-7)
        self.assertTrue(np.any(x == 0.3))
        self.assertTrue(np.any(x == -0.2))

    def test_refinement_reduces_train_error_and_freezes_memory(self):
        t = np.arange(1000) / 64
        u = 0.03 * np.sin(t * 1.2)
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
        spec = dict(
            family="elman",
            hidden=2,
            mean=[0] * 5,
            scale=[1] * 5,
            head=[0.2, 0.1],
            bias=0.005,
            wi=[[0.2, 0.1, 0, 0, 0.1], [0.1, 0.2, 0, 0, 0.2]],
            wh=[[0.4, 0], [0, 0.2]],
            bi=[0.01, -0.01],
            bh=[0, 0],
            rate_scale=1,
            rate_limit=3,
            residual=True,
            gate=True,
        )
        truth = dict(
            ddv=dict(scale=1, lags=0, basis="legacy", weights=[0, 1, 0, 0, 0]),
            angle=dict(scale=1, weights=[0, 1]),
            ram=dict(
                kind="sequence_rate", reference=ref, bounds=[-3, 3], sequence=spec
            ),
        )
        y = rollout(truth, u, 0)[1]
        model = copy.deepcopy(truth)
        model["ram"]["sequence"]["head"] = [0, 0]
        model["ram"]["sequence"]["bias"] = 0
        flight = Flight("a", t, np.c_[u, u, y, y])
        improved = refine_sequence_head(model, [flight], iterations=3, ridge=0.001)
        before = np.mean((rollout(model, u, 0)[1] - y) ** 2)
        after = np.mean((rollout(improved, u, 0)[1] - y) ** 2)
        self.assertLess(after, before * 0.1)
        self.assertEqual(improved["ram"]["sequence"]["wh"], spec["wh"])
        self.assertEqual(model["ram"]["sequence"]["head"], [0, 0])

    def test_goal_is_per_flight_and_strict(self):
        self.assertFalse(meets_goal(89.99, 99))
        self.assertFalse(meets_goal(95, 75))
        self.assertFalse(meets_goal(90, 80))
        self.assertTrue(meets_goal(90.01, 75.01))
        self.assertFalse(meets_goal(float("nan"), 90))

    def test_training_weight_validation(self):
        from actuator.sequence_train import fit_sequence

        arrays = [
            (np.zeros((160, 5)), np.zeros(160), np.zeros(160), np.ones(160, dtype=bool))
        ] * 2
        norm = dict(mean=[0] * 5, scale=[1] * 5, rate_scale=1, rate_limit=1)
        for weights in ([1], [1, -1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                fit_sequence((arrays, norm), {}, flight_weights=weights, horizon=96)
        with self.assertRaises(ValueError):
            fit_sequence((arrays, norm), {}, position_weight=-1, horizon=96)
