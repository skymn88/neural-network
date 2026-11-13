"""TRAIN-only sequence-rate fitting; export explicit NumPy inference weights."""

import copy
import numpy as np
from .data import FS
from .inference import predict_ddv
from .sequence_inference import sequence_features, temporal_features


def prepare_sequences(train, reference):
    arrays = []
    bounds = np.asarray(reference["ram"]["bounds"])
    span = np.ptp(bounds)
    for f in train:
        z = predict_ddv(reference["ddv"], f.values[:, 0])
        features, base = sequence_features(z, reference["ram"])
        x = f.values[:, 2]
        velocity = np.r_[0, np.diff(x) * FS]
        valid = (
            (x > bounds[0] + 0.015 * span)
            & (x < bounds[1] - 0.015 * span)
            & (np.arange(len(x)) > 32)
        )
        arrays.append((features, base, velocity, valid))
    all_f = np.concatenate([a[0] for a in arrays])
    scale = np.maximum(np.std(all_f, axis=0), 1e-5)
    mean = np.mean(all_f, axis=0)
    v = np.concatenate([a[2][a[3]] for a in arrays])
    rate_scale = max(float(np.std(v)), 1.0)
    limit = float(np.quantile(abs(v), 0.999) * 1.2)
    return arrays, dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        rate_scale=rate_scale,
        rate_limit=limit,
    )


def fit_sequence(
    prepared,
    reference,
    family="gru",
    hidden=16,
    residual=True,
    gate=True,
    steps=400,
    horizon=256,
    seed=42,
    ridge=0.001,
    leak=0.2,
    flight_weights=None,
    position_weight=0.2,
):
    if family not in ("gru", "lstm", "elman", "tdnn", "tcn", "esn", "rbf"):
        raise ValueError("Unsupported sequence family")
    if horizon < 96 or steps < 1 or hidden < 1:
        raise ValueError(
            "Sequence training requires horizon >=96, steps >=1, hidden >=1"
        )
    if any(len(a[0]) <= horizon + 34 for a in prepared[0]):
        raise ValueError("TRAIN segments must exceed horizon+34 samples")
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arrays, norm = prepared
    probabilities = None
    if flight_weights is not None:
        weights = np.asarray(flight_weights, dtype=float)
        if (
            weights.shape != (len(arrays),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "flight_weights must be positive finite weights per TRAIN flight"
            )
        probabilities = weights / weights.sum()
    if not np.isfinite(position_weight) or position_weight < 0:
        raise ValueError("position_weight must be finite and nonnegative")

    spec = dict(norm, family=family, hidden=hidden, residual=residual, gate=gate)
    normalized = [
        (a - np.asarray(norm["mean"])) / np.asarray(norm["scale"])
        for a, _, _, _ in arrays
    ]
    offsets = [0, 1, 4, 16, 64, 256]
    if family in ("esn", "rbf"):
        if family == "rbf":
            X = np.concatenate(
                [temporal_features(x, offsets)[32::8] for x in normalized]
            )
            valid = np.concatenate([a[3][32::8] for a in arrays])
            Y = (
                np.concatenate(
                    [(a[2] - (a[1] if residual else 0))[32::8] for a in arrays]
                )
                / norm["rate_scale"]
            )
            X, Y = X[valid], Y[valid]
            centers = X[rng.choice(len(X), hidden, replace=False)]
            width2 = max(
                float(np.median(np.sum((centers[1:] - centers[:-1]) ** 2, axis=1))), 1.0
            )
            spec.update(centers=centers.tolist(), width2=width2, offsets=offsets)
            G = np.zeros((hidden + 1, hidden + 1))
            b = np.zeros(hidden + 1)
            for start in range(0, len(X), 2048):
                x = X[start : start + 2048]
                dist = np.maximum(
                    (x * x).sum(1)[:, None]
                    + (centers * centers).sum(1)[None, :]
                    - 2 * x @ centers.T,
                    0,
                )
                A = np.column_stack([np.exp(-dist / width2), np.ones(len(x))])
                G += A.T @ A
                b += A.T @ Y[start : start + len(x)]
            n = len(X)
        else:
            wi = rng.normal(0, 0.3, (hidden, 5))
            wh = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden))
            wh *= 0.9 / max(abs(np.linalg.eigvals(wh)))
            bi = rng.normal(0, 0.05, hidden)
            spec.update(
                wi=wi.tolist(),
                wh=wh.tolist(),
                bi=bi.tolist(),
                bh=np.zeros(hidden).tolist(),
                leak=leak,
            )
            G = np.zeros((hidden + 1, hidden + 1))
            b = np.zeros(hidden + 1)
            n = 0
            for x, (_, base, v, valid) in zip(normalized, arrays):
                h = np.zeros(hidden)
                rows = []
                targets = []
                for k, inp in enumerate(x @ wi.T + bi):
                    h = (1 - leak) * h + leak * np.tanh(inp + wh @ h)
                    if k % 8 == 0 and valid[k]:
                        rows.append(np.r_[h, 1])
                        targets.append(
                            (v[k] - (base[k] if residual else 0)) / norm["rate_scale"]
                        )
                A = np.asarray(rows)
                y = np.asarray(targets)
                G += A.T @ A
                b += A.T @ y
                n += len(y)
        weights = np.linalg.solve(G / n + ridge * np.eye(hidden + 1), b / n)
        spec.update(head=weights[:-1].tolist(), bias=float(weights[-1]), ridge=ridge)
    else:
        import torch.nn.functional as F

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if family in ("gru", "lstm", "elman"):
                    cls = {
                        "gru": torch.nn.GRU,
                        "lstm": torch.nn.LSTM,
                        "elman": torch.nn.RNN,
                    }[family]
                    self.core = cls(5, hidden, batch_first=True)
                elif family == "tdnn":
                    self.core = torch.nn.Linear(5 * len(offsets), hidden)
                elif family == "tcn":
                    self.c1 = torch.nn.Conv1d(5, hidden, 5, dilation=4)
                    self.c2 = torch.nn.Conv1d(hidden, hidden, 5, dilation=16)
                else:
                    raise ValueError("Unsupported family")
                self.head = torch.nn.Linear(hidden, 1)
                torch.nn.init.zeros_(self.head.weight)
                torch.nn.init.zeros_(self.head.bias)

            def forward(self, x):
                if family in ("gru", "lstm", "elman"):
                    h = self.core(x)[0]
                elif family == "tdnn":
                    h = torch.tanh(self.core(x))
                else:
                    h = torch.tanh(self.c1(F.pad(x.transpose(1, 2), (16, 0))))
                    h = torch.tanh(self.c2(F.pad(h, (64, 0)))).transpose(1, 2)
                return self.head(h).squeeze(-1)

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=0.002)
        tensors = []
        events = []
        for x, (feat, base, v, valid) in zip(normalized, arrays):
            if family == "tdnn":
                x = temporal_features(x, offsets)
            tensors.append(
                tuple(
                    torch.tensor(a, dtype=torch.float32)
                    for a in (x, base, v, valid.astype(float))
                )
            )
            events.append(
                np.flatnonzero(
                    valid
                    & (abs(v) > 1)
                    & (np.arange(len(v)) > horizon)
                    & (np.arange(len(v)) < len(v) - horizon)
                )
            )
        for step in range(steps):
            batch = []
            for j in range(12):
                fi = (
                    int(rng.integers(len(tensors)))
                    if probabilities is None
                    else int(rng.choice(len(tensors), p=probabilities))
                )
                item = tensors[fi]
                if j % 2 and len(events[fi]):
                    start = int(rng.choice(events[fi])) - int(
                        rng.integers(horizon // 2)
                    )
                else:
                    start = int(rng.integers(33, len(item[0]) - horizon))
                batch.append([a[start : start + horizon] for a in item])
            X, B, Y, M = [torch.stack([a[k] for a in batch]) for k in range(4)]
            pred = net(X) * norm["rate_scale"] + (B if residual else 0)
            if gate:
                pred = pred * (B != 0)
            pred = pred.clamp(-norm["rate_limit"], norm["rate_limit"])
            # First 64 samples warm recurrent/conv state; no reset in evaluation.
            err = (pred[:, 64:] - Y[:, 64:]) / norm["rate_scale"]
            mask = M[:, 64:]
            speed = (err.square() * mask).sum() / mask.sum().clamp(min=1)
            interior = (mask.min(dim=1).values > 0).float()
            pos = (err.cumsum(dim=1) / FS).square().mean(dim=1)
            loss = speed + position_weight * (
                pos * interior
            ).sum() / interior.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
        spec.update(
            head=net.head.weight.detach().numpy().ravel().tolist(),
            bias=float(net.head.bias.detach()[0]),
            steps=steps,
            horizon=horizon,
            flight_weights=flight_weights,
            position_weight=position_weight,
        )
        if family in ("gru", "lstm", "elman"):
            spec.update(
                {
                    k: v.detach().numpy().tolist()
                    for k, v in dict(
                        wi=net.core.weight_ih_l0,
                        wh=net.core.weight_hh_l0,
                        bi=net.core.bias_ih_l0,
                        bh=net.core.bias_hh_l0,
                    ).items()
                }
            )
        elif family == "tdnn":
            spec.update(
                w=net.core.weight.detach().numpy().tolist(),
                b=net.core.bias.detach().numpy().tolist(),
                offsets=offsets,
            )
        else:
            spec["layers"] = [
                dict(
                    w=c.weight.detach().numpy().tolist(),
                    b=c.bias.detach().numpy().tolist(),
                    dilation=d,
                )
                for c, d in [(net.c1, 4), (net.c2, 16)]
            ]
    model = copy.deepcopy(reference)
    model["ram"] = dict(
        kind="sequence_rate",
        sequence=spec,
        reference=reference["ram"],
        bounds=reference["ram"]["bounds"],
    )
    return model
