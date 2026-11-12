"""Deterministic contract checks; small generated fixtures, never real-flight claims."""

import base64
import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.signal import lfilter
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from actuator import data, models, experiment, inference, server


def fixture(n=4, samples=512):
    flights = []
    for j in range(n):
        t = np.arange(samples) / 64
        u = 0.2 * np.sin(t * (1 + j * 0.1)) + 0.05 * np.cos(3 * t)
        z = lfilter([0.25], [1, -0.75], u)
        x = j + np.cumsum(z) / 64
        flights.append(data.Flight(f"f{j}", t, np.c_[u, z, x, 0.3 * x]))
    return flights


class DataTests(unittest.TestCase):
    def test_split_counts(self):
        for n, counts in [
            (4, (2, 1, 1)),
            (9, (7, 1, 1)),
            (10, (7, 2, 1)),
            (11, (7, 2, 2)),
            (20, (14, 4, 2)),
        ]:
            groups = data.split(fixture(n))
            self.assertEqual(tuple(map(len, groups.values())), counts)
            names = [f.name for fs in groups.values() for f in fs]
            self.assertEqual(len(set(names)), n)

    def test_single_flight_guard(self):
        groups = data.split(fixture(1, 4096))
        a, b, c = [groups[s][0] for s in groups]
        self.assertGreater(b.time[0] - a.time[-1], 1.9)
        self.assertGreater(c.time[0] - b.time[-1], 1.9)

    def test_explicit_split_rejects_unknown(self):
        with self.assertRaises(ValueError):
            data.split(fixture(), assignments=["train", "validation", "test", "typo"])

    def test_antialias(self):
        t = np.arange(4096) / 256
        high = np.sin(2 * np.pi * 70 * t)
        low = np.sin(2 * np.pi * 2 * t)
        f = data.align("high", [(t, high)] * 4)
        g = data.align("low", [(t, low)] * 4)
        self.assertLess(np.std(f.values[32:-32, 0]), 0.02)
        self.assertGreater(np.std(g.values[32:-32, 0]), 0.69)
        self.assertAlmostEqual(np.mean(np.diff(f.time)), 1 / 64)

    def test_gap_and_missing_signal(self):
        t = np.r_[np.arange(100) / 64, 10 + np.arange(100) / 64]
        with self.assertRaises(ValueError):
            data.align("gap", [(t, t)] * 4)
        with self.assertRaises(ValueError):
            data.align("missing", [(t, t)] * 3)

    def test_original_sha(self):
        self.assertEqual(
            hashlib.sha256(
                (data.ROOT / "data/synthetic/synthetic_rich.mat").read_bytes()
            ).hexdigest(),
            "22fab73b38de0f4b40da981f21f10bf55b02016da40b13ec7f6b4fb09a75c8fb",
        )

    def test_pulse_reproducibility(self):
        a, _ = data.pulse_flights(durations=(6, 7, 8, 9))
        b, _ = data.pulse_flights(durations=(6, 7, 8, 9))
        self.assertEqual(data.fingerprint(a), data.fingerprint(b))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = fixture(5)
        cls.train = cls.f[:3]
        cls.val = cls.f[3:4]
        cls.ddv, _ = models.fit_ddv(cls.train, cls.val, length=4)
        cls.angle = models.fit_angle(cls.train, cls.val)

    def test_ddv_normal_equations(self):
        m, _ = models.fit_ddv(self.train, self.val, length=4, fixed_ridge=1e-5)
        A = np.concatenate(
            [
                inference.ddv_features(f.values[:, 0], m["scale"], 4, "rich")[32:]
                for f in self.train
            ]
        )
        y = np.concatenate([f.values[32:, 1] for f in self.train])
        R = np.eye(A.shape[1])
        R[0, 0] = 0
        dense = np.linalg.solve(A.T @ A / len(y) + 1e-5 * R, A.T @ y / len(y))
        np.testing.assert_allclose(m["weights"], dense, rtol=1e-7, atol=1e-8)

    def test_oof_excludes_held_flight(self):
        with patch.object(models, "fit_ddv", wraps=models.fit_ddv) as fit:
            models.fit_oe(self.train, self.val, self.ddv, self.angle)
        self.assertEqual(fit.call_count, 3)
        for k, call in enumerate(fit.call_args_list):
            self.assertNotIn(self.train[k].name, [f.name for f in call.args[0]])

    def test_portable_oe_matches_filter(self):
        m, _ = models.fit_oe(self.train, self.val, self.ddv, self.angle)
        f = self.val[0]
        z, x, _ = inference.rollout(m, f.values[:, 0], f.values[0, 2])
        r = m["ram"]
        d = inference.ram_features(z, r["scale"], r["lags"]) @ r["weights"]
        d[0] = 0
        a = 1 + r["beta"] / 64
        expected = f.values[0, 2] * a ** np.arange(len(d)) + lfilter(
            [1 / 64], [1, -a], d
        )
        np.testing.assert_allclose(x, expected, atol=1e-11)
        clone = json.loads(json.dumps(m))
        np.testing.assert_array_equal(
            inference.rollout(clone, f.values[:, 0], f.values[0, 2])[1], x
        )

    def test_neural_numpy_torch_parity(self):
        import torch

        rng = np.random.default_rng(4)
        W1 = rng.normal(0, 0.01, (5, 3))
        b1 = rng.normal(0, 0.01, 5)
        W2 = rng.normal(0, 0.01, 5)
        m = {
            "ddv": self.ddv,
            "angle": self.angle,
            "ram": dict(
                kind="neural",
                ddv_scale=1.0,
                x_mean=0.0,
                x_scale=1.0,
                v_scale=1.0,
                W1=W1.tolist(),
                b1=b1.tolist(),
                W2=W2.tolist(),
                b2=0.001,
            ),
        }
        u = self.val[0].values[:100, 0]
        z, x, _ = inference.rollout(m, u, 2.0)
        xt = torch.tensor(2.0, dtype=torch.float64)
        vt = torch.tensor(0.0, dtype=torch.float64)
        expected = [2.0]
        for v in z[1:]:
            q = torch.stack([torch.tensor(v), xt, vt])
            vt = (
                vt
                + torch.tensor(W2) @ torch.tanh(torch.tensor(W1) @ q + torch.tensor(b1))
                + 0.001
            )
            xt = xt + vt / 64
            expected.append(float(xt))
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_causal_predictions(self):
        u = self.val[0].values[:, 0].copy()
        a = inference.predict_ddv(self.ddv, u)
        u[200:] += 100
        b = inference.predict_ddv(self.ddv, u)
        np.testing.assert_array_equal(a[:200], b[:200])

    def test_constant_fit_undefined(self):
        self.assertIsNone(experiment.metrics(np.ones(100), np.ones(100))["fit"])

    def test_train_smoke_and_test_isolation(self):
        f = fixture(4)
        groups = {"train": f[:2], "validation": f[2:3], "test": f[3:]}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(experiment, "ROOT", Path(tmp)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            a, ma, out = experiment.run(
                f,
                groups,
                source="fixture",
                methods=["oe", "neural"],
                steps=4,
                horizon=16,
                hidden=4,
                warm_steps=5,
                log=lambda _: None,
            )
            self.assertFalse(any(r["split"] == "test" for r in a["per_flight"]))
            self.assertEqual(
                load_workbook(out / "comparison.xlsx", read_only=True).sheetnames,
                ["Summary", "PerFlight", "Search", "Diagnostics", "Config"],
            )
            f[-1].values[:] = 999999
            b, mb, _ = experiment.run(
                f,
                groups,
                source="fixture",
                methods=["oe", "neural"],
                steps=4,
                horizon=16,
                hidden=4,
                warm_steps=5,
                log=lambda _: None,
            )
            self.assertEqual(a["validation_scores"], b["validation_scores"])
            self.assertEqual(ma, mb)
            json.dumps(a, allow_nan=False)

    def test_short_window_message(self):
        with self.assertRaisesRegex(ValueError, "horizon"):
            models.fit_neural(self.train, self.val, self.ddv, self.angle, horizon=512)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_page_and_origin(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(
            self.client.post(
                "/api/run", headers={"Origin": "https://foreign.example"}, json={}
            ).status_code,
            403,
        )

    def test_csv_import(self):
        t = np.arange(2048) / 64
        rows = ["time,CoilCurrent,DDV,RAM,Deflection"] + [
            f"{a},{np.sin(a)},{np.sin(a)},0,0" for a in t
        ]
        file = {
            "name": "flight.csv",
            "content": base64.b64encode("\n".join(rows).encode()).decode(),
        }
        self.assertEqual(
            self.client.post("/api/inspect", json={"file": file}).json()["rows"], 2048
        )
        body = {
            "files": [file],
            "mapping": {k: k for k in data.NAMES},
            "times": {k: "time" for k in data.NAMES},
        }
        r = self.client.post("/api/import", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        body["mapping"]["DDV"] = "missing"
        self.assertEqual(self.client.post("/api/import", json=body).status_code, 400)

    def test_invalid_run_and_artifact(self):
        self.assertEqual(
            self.client.post("/api/run", json={"source": "user"}).status_code, 400
        )
        self.assertEqual(
            self.client.post("/api/run", json={"methods": ["bad"]}).status_code, 422
        )
        self.assertEqual(self.client.get("/api/results/abc").status_code, 400)
        self.assertEqual(
            self.client.get("/api/results/1/files/private.csv").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
