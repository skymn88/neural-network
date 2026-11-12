"""Learned nonlinear rate map, with an optional causal neutral observer.
No simulator Truth or physical gains are read. Position bounds, neutral gate,
static map and sign are estimated from training measurements only.
"""

import numpy as np
from .data import FS
from .inference import predict_ddv, neutral_features, lags


def fit_null_rate(
    train,
    ddv_model,
    angle,
    adaptive=False,
    knots=81,
    tau=0.5,
    delay=1,
    ridge=1e-5,
    observer="legacy",
):
    if tau <= 0 or knots < 5 or delay < 0 or int(delay) != delay:
        raise ValueError(
            "Positive tau, >=5 knots and nonnegative integer delay required"
        )
    all_x = np.concatenate([f.values[:, 2] for f in train])
    bounds = np.quantile(all_x, [0.0001, 0.9999])
    span = np.ptp(bounds)
    hold = []
    for f in train:
        # Interior stationary samples avoid falsely treating parked spool as null.
        x = f.values[:, 2]
        v = np.r_[0, np.diff(x) * FS]
        mask = (
            (abs(v) < max(span * 0.001, 0.05))
            & (x > bounds[0] + 0.03 * span)
            & (x < bounds[1] - 0.03 * span)
        )
        hold.extend(f.values[mask, 1])
    if len(hold) < 64:
        raise ValueError("Not enough interior hold data to identify neutral DDV")
    hold = np.asarray(hold)
    q = np.quantile(hold, [0.001, 0.999])
    margin = max(0.05 * np.ptp(q), 0.002)
    spec = {
        "center": float(np.median(hold)),
        "gate": [float(q[0] - margin), float(q[1] + margin)],
        "tau": tau,
        "max_step": float(max(np.ptp(q) * 0.1, 0.004)),
        "adaptive": adaptive,
    }
    if observer not in ("legacy", "stationary", "envelope"):
        raise ValueError("Unknown neutral observer")
    if observer != "legacy":
        ranges = []
        for f in train:
            z = predict_ddv(ddv_model, f.values[:, 0])
            x = f.values[:, 2]
            v = np.r_[0, np.diff(x) * FS]
            mask = (
                (abs(v) < max(span * 0.001, 0.05))
                & (x > bounds[0] + 0.03 * span)
                & (x < bounds[1] - 0.03 * span)
            )
            ranges.extend(np.ptp(lags(z, 31), axis=1)[mask])
        spec.update(
            mode=observer,
            quiet_window=32,
            quiet_range=float(max(np.quantile(ranges, 0.95), 1e-6)),
            envelope_age=8 * FS,
        )
    hold_inputs = []
    inputs = []
    targets = []
    for f in train:
        # Predicted upstream DDV, not generator internals. Current/DDV fit is supervised.
        z = predict_ddv(ddv_model, f.values[:, 0])
        u, _ = neutral_features(z, spec)
        u = np.r_[np.repeat(u[0], delay), u[: len(u) - delay]] if delay else u
        x = f.values[:, 2]
        v = np.r_[0, np.diff(x) * FS]
        interior = (x > bounds[0] + 0.015 * span) & (x < bounds[1] - 0.015 * span)
        valid = interior & (np.arange(len(x)) > 32)
        if observer != "legacy":
            quiet = np.ptp(lags(z, 31), axis=1) <= spec["quiet_range"]
            hold_inputs.extend(u[valid & quiet & (abs(v) < max(span * 0.001, 0.05))])
        inputs.append(u[valid])
        targets.append(v[valid])
    u = np.concatenate(inputs)
    v = np.concatenate(targets)
    # Equally spaced learned-input coverage; coefficients use only TRAIN targets.
    grid = np.linspace(np.quantile(u, 0.0001), np.quantile(u, 0.9999), knots)
    index = np.clip(np.searchsorted(grid, u) - 1, 0, knots - 2)
    frac = np.clip((u - grid[index]) / (grid[index + 1] - grid[index]), 0, 1)
    # Inverse occupancy balances rare motion against hundreds of seconds of holds.
    counts = np.bincount(index, minlength=knots)
    w = 1 / np.maximum(counts[index], 1)
    G = np.zeros((knots, knots))
    h = np.zeros(knots)
    np.add.at(G, (index, index), w * (1 - frac) ** 2)
    np.add.at(G, (index + 1, index + 1), w * frac**2)
    np.add.at(G, (index, index + 1), w * frac * (1 - frac))
    np.add.at(G, (index + 1, index), w * frac * (1 - frac))
    np.add.at(h, index, w * (1 - frac) * v)
    np.add.at(h, index + 1, w * frac * v)
    D = np.diff(np.eye(knots), n=2, axis=0)
    values = np.linalg.solve(G + ridge * (D.T @ D) + 1e-10 * np.eye(knots), h)
    # Neutral plateau learnt from median speed versus centered input.
    hold_rate = max(span * 0.002, 0.15)
    near = (abs(grid) < np.ptp(grid) * 0.1) & (abs(values) < hold_rate)
    if np.any(near):
        ids = np.flatnonzero(near)
        values[ids] = 0
    model = {
        "ddv": ddv_model,
        "angle": angle,
        "ram": {
            "kind": "null_rate",
            "observer": spec,
            "knots": grid.tolist(),
            "values": values.tolist(),
            "bounds": bounds.tolist(),
            "delay": delay,
            "ridge": ridge,
            "training_rows": len(u),
        },
    }

    if observer != "legacy" and hold_inputs:
        band = np.quantile(hold_inputs, [0.01, 0.99])
        model["ram"]["zero_band"] = band.tolist()
        inside = (grid >= band[0]) & (grid <= band[1])
        values[inside] = 0.0
        model["ram"]["values"] = values.tolist()
    return model


def refine_window_rate(model, train, seconds=5.0, ridge=0.01):
    """TRAIN window position-error refinement; no rollout resets at inference."""
    import copy

    m = copy.deepcopy(model)
    r = m["ram"]
    grid = np.asarray(r["knots"])
    prior = np.asarray(r["values"])
    p = len(grid)
    window = int(seconds * FS)
    G = np.zeros((p, p))
    h = np.zeros(p)
    n = 0
    lo, hi = r["bounds"]
    span = hi - lo
    for f in train:
        z = predict_ddv(m["ddv"], f.values[:, 0])
        u, _ = neutral_features(z, r["observer"])
        delay = r["delay"]
        if delay:
            u = np.r_[np.repeat(u[0], delay), u[:-delay]]
        index = np.clip(np.searchsorted(grid, u) - 1, 0, p - 2)
        frac = np.clip((u - grid[index]) / (grid[index + 1] - grid[index]), 0, 1)
        Z = np.zeros((len(u), p))
        Z[np.arange(len(u)), index] = 1 - frac
        Z[np.arange(len(u)), index + 1] = frac
        band = r["zero_band"]
        Z[(u >= band[0]) & (u <= band[1])] = 0
        size = ((len(u) - 33) // window) * window
        x = f.values[:, 2]
        chunks = x[33 : 33 + size].reshape(-1, window)
        starts = x[32 : 32 + size : window]
        valid = (
            (chunks.min(axis=1) > lo + 0.015 * span)
            & (chunks.max(axis=1) < hi - 0.015 * span)
            & (starts > lo + 0.015 * span)
            & (starts < hi - 0.015 * span)
        )
        A = np.cumsum(Z[33 : 33 + size].reshape(-1, window, p)[valid], axis=1) / FS
        Y = chunks[valid] - starts[valid, None]
        A = A.reshape(-1, p)
        Y = Y.ravel()
        G += A.T @ A
        h += A.T @ Y
        n += len(Y)
    if not n:
        raise ValueError("No unclipped TRAIN windows for refinement")
    scale = np.sqrt(np.maximum(np.diag(G) / n, 1e-10))
    Q = G / np.outer(scale, scale) / n
    delta = np.linalg.solve(Q + ridge * np.eye(p), (h - G @ prior) / scale / n) / scale
    r["values"] = (prior + delta).tolist()
    r["window_refinement"] = {"seconds": seconds, "ridge": ridge, "rows": n}
    return m
