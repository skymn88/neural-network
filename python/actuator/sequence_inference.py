"""NumPy-only finite-state sequence-rate models; no measured feedback or Truth."""

import numpy as np


def sequence_features(ddv, reference):
    from .inference import neutral_features

    u, center = neutral_features(ddv, reference["observer"])
    delay = reference["delay"]
    if delay:
        u = np.r_[np.repeat(u[0], delay), u[:-delay]]
    rate = np.interp(u, reference["knots"], reference["values"])
    if "map_rate_limit" in reference:
        rate = np.clip(rate, -reference["map_rate_limit"], reference["map_rate_limit"])
    if "zero_band" in reference:
        lo, hi = reference["zero_band"]
        rate[(u >= lo) & (u <= hi)] = 0
    delta = np.r_[0.0, np.diff(ddv)]
    return np.column_stack([ddv, u, center, delta, rate]), rate


def temporal_features(features, offsets):
    n = len(features)
    return np.column_stack([features[np.maximum(np.arange(n) - k, 0)] for k in offsets])


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def sequence_raw(spec, features):
    """Return unscaled network/readout output. Recurrent state starts at zero."""
    x = (features - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
    family = spec["family"]
    head = np.asarray(spec["head"])
    if family in ("tdnn", "rbf"):
        x = temporal_features(x, spec["offsets"])
        if family == "tdnn":
            h = np.tanh(x @ np.asarray(spec["w"]).T + np.asarray(spec["b"]))
        else:
            # Chunked distances prevent a full flight x centers x input tensor.
            centers = np.asarray(spec["centers"])
            out = np.empty(len(x))
            for start in range(0, len(x), 2048):
                a = x[start : start + 2048]
                dist = np.maximum(
                    (a * a).sum(1)[:, None]
                    + (centers * centers).sum(1)[None, :]
                    - 2 * a @ centers.T,
                    0,
                )
                out[start : start + len(a)] = (
                    np.exp(-dist / spec["width2"]) @ head + spec["bias"]
                )
            return out
        return h @ head + spec["bias"]
    if family == "tcn":
        # Two causal, dilated conv/tanh layers; zero padding matches torch training.
        for layer in spec["layers"]:
            w = np.asarray(layer["w"])
            y = np.broadcast_to(np.asarray(layer["b"]), (len(x), len(w))).copy()
            for j in range(w.shape[2]):
                lag = (w.shape[2] - 1 - j) * layer["dilation"]
                if lag:
                    y[lag:] += x[:-lag] @ w[:, :, j].T
                else:
                    y += x @ w[:, :, j].T
            x = np.tanh(y)
        return x @ head + spec["bias"]
    h = np.zeros(spec["hidden"])
    c = h.copy()
    out = np.empty(len(x))
    wi, wh = np.asarray(spec["wi"]), np.asarray(spec["wh"])
    bi, bh = np.asarray(spec["bi"]), np.asarray(spec["bh"])
    # Input projection is vectorized; recurrence has fixed-size state.
    projected = x @ wi.T + bi
    for k, inp in enumerate(projected):
        rec = wh @ h + bh
        if family == "gru":
            ir, iz, inn = np.split(inp, 3)
            hr, hz, hn = np.split(rec, 3)
            reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
            new = np.tanh(inn + reset * hn)
            h = (1 - update) * new + update * h
        elif family == "lstm":
            i, f, g, o = np.split(inp + rec, 4)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
        elif family == "esn":
            h = (1 - spec["leak"]) * h + spec["leak"] * np.tanh(inp + rec)
        else:
            h = np.tanh(inp + rec)
        out[k] = h @ head + spec["bias"]
    return out


def combine_rate(spec, raw, base_rate):
    """Shared clipping/gating policy for portable and accelerated inference."""
    rate = raw * spec["rate_scale"]
    if "correction_limit" in spec:
        rate = np.clip(rate, -spec["correction_limit"], spec["correction_limit"])
    if spec["residual"]:
        rate = base_rate + rate
    if spec["gate"]:
        rate[base_rate == 0] = 0
    limit = spec.get("output_rate_limit", spec["rate_limit"])
    return np.clip(rate, -limit, limit)


def sequence_rate(spec, features, base_rate):
    return combine_rate(spec, sequence_raw(spec, features), base_rate)


def integrate_rate(rate, initial, bounds):
    x = np.empty(len(rate))
    x[0] = initial
    lo, hi = bounds
    for k in range(1, len(rate)):
        x[k] = min(hi, max(lo, x[k - 1] + rate[k] / 64))
    return x


def ensemble_rate(ram, ddv):
    rates = []
    for member in ram["members"]:
        features, base = sequence_features(ddv, member["reference"])
        rates.append(sequence_rate(member["sequence"], features, base))
    return np.mean(rates, axis=0)
