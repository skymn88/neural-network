import unittest
import numpy as np
import torch
from actuator.sequence_inference import (
    sequence_raw,
    sequence_rate,
    temporal_features,
    sequence_features,
)
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.data import Flight


class SequenceTests(unittest.TestCase):
    def test_recurrent_torch_numpy_parity_and_causality(self):
        torch.manual_seed(7)
        x = np.random.default_rng(7).normal(size=(71, 5)).astype("float32")
        for family, cls in [
            ("gru", torch.nn.GRU),
            ("lstm", torch.nn.LSTM),
            ("elman", torch.nn.RNN),
        ]:
            core = cls(5, 4, batch_first=True)
            head = torch.nn.Linear(4, 1)
            spec = dict(
                family=family,
                hidden=4,
                mean=[0] * 5,
                scale=[1] * 5,
                head=head.weight.detach().numpy().ravel(),
                bias=float(head.bias.detach()[0]),
            )
            spec.update(
                {
                    k: v.detach().numpy()
                    for k, v in dict(
                        wi=core.weight_ih_l0,
                        wh=core.weight_hh_l0,
                        bi=core.bias_ih_l0,
                        bh=core.bias_hh_l0,
                    ).items()
                }
            )
            target = head(core(torch.tensor(x)[None])[0]).detach().numpy().ravel()
            pred = sequence_raw(spec, x)
            np.testing.assert_allclose(pred, target, atol=2e-7)
            np.testing.assert_allclose(
                sequence_raw(spec, x[:30]), pred[:30], atol=1e-14
            )

    def test_tcn_and_tdnn_parity(self):
        import torch.nn.functional as F

        torch.manual_seed(9)
        x = np.random.default_rng(9).normal(size=(95, 5)).astype("float32")
        head = torch.nn.Linear(4, 1)
        spec = dict(
            mean=[0] * 5,
            scale=[1] * 5,
            head=head.weight.detach().numpy().ravel(),
            bias=float(head.bias.detach()[0]),
        )
        c1 = torch.nn.Conv1d(5, 4, 5, dilation=4)
        c2 = torch.nn.Conv1d(4, 4, 5, dilation=16)
        h = torch.tanh(c1(F.pad(torch.tensor(x).T[None], (16, 0))))
        y = (
            head(torch.tanh(c2(F.pad(h, (64, 0)))).transpose(1, 2))
            .detach()
            .numpy()
            .ravel()
        )
        tcn = dict(
            spec,
            family="tcn",
            layers=[
                dict(w=c.weight.detach().numpy(), b=c.bias.detach().numpy(), dilation=d)
                for c, d in [(c1, 4), (c2, 16)]
            ],
        )
        np.testing.assert_allclose(sequence_raw(tcn, x), y, atol=2e-7)
        np.testing.assert_allclose(
            sequence_raw(tcn, x[:30]), sequence_raw(tcn, x)[:30], atol=1e-14
        )
        offsets = [0, 2, 5]
        linear = torch.nn.Linear(15, 4)
        tdnn = dict(
            spec,
            family="tdnn",
            offsets=offsets,
            w=linear.weight.detach().numpy(),
            b=linear.bias.detach().numpy(),
        )
        y = (
            head(torch.tanh(linear(torch.tensor(temporal_features(x, offsets)))))
            .detach()
            .numpy()
            .ravel()
        )
        np.testing.assert_allclose(sequence_raw(tdnn, x), y, atol=2e-7)

    def test_fit_smoke_all_families_and_hold_gate(self):
        n = 600
        t = np.arange(n) / 64
        z = 0.04 * np.sin(t * 2)
        ref = dict(
            ddv=dict(scale=1, lags=0, basis="legacy", weights=[0, 1, 0, 0, 0]),
            angle=dict(scale=1, weights=[0, 1]),
            ram=dict(
                kind="null_rate",
                bounds=[-2, 2],
                knots=[-0.1, 0, 0.1],
                values=[-1, 0, 1],
                delay=1,
                zero_band=[-0.001, 0.001],
                observer=dict(
                    center=0, gate=[-0.01, 0.01], tau=2, max_step=0.01, adaptive=False
                ),
            ),
        )
        flights = [Flight("a", t, np.c_[z, z, 0.1 * np.sin(t), 0.1 * np.sin(t)])]
        prepared = prepare_sequences(flights, ref)
        self.assertEqual(prepared[0][0][0].shape, (n, 5))
        features, base = sequence_features(z, ref["ram"])
        for family in ("gru", "lstm", "elman", "tdnn", "tcn", "esn", "rbf"):
            m = fit_sequence(
                prepared, ref, family=family, hidden=4, steps=2, horizon=96
            )
            pred = sequence_rate(m["ram"]["sequence"], features, base)
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(np.all(pred[base == 0] == 0))
            np.testing.assert_allclose(
                sequence_features(z[:30], ref["ram"])[0], features[:30], atol=1e-14
            )
