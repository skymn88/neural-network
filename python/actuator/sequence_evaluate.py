"""Optional Torch-accelerated scoring, same exported model and full-flight state."""

import numpy as np
from .inference import rollout, predict_ddv, angle_values
from .sequence_inference import sequence_features


def fast_rollout(model, current, initial):
    r = model["ram"]
    if r["kind"] != "sequence_rate" or r["sequence"]["family"] not in (
        "gru",
        "lstm",
        "elman",
    ):
        return rollout(model, current, initial)
    import torch

    torch.set_num_threads(1)
    spec = r["sequence"]
    family = spec["family"]
    cls = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM, "elman": torch.nn.RNN}[family]
    net = cls(5, spec["hidden"], batch_first=True).double()
    for attr, key in [
        ("weight_ih_l0", "wi"),
        ("weight_hh_l0", "wh"),
        ("bias_ih_l0", "bi"),
        ("bias_hh_l0", "bh"),
    ]:
        getattr(net, attr).data.copy_(torch.tensor(spec[key], dtype=torch.float64))
    z = predict_ddv(model["ddv"], current)
    features, base = sequence_features(z, r["reference"])
    features = (features - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
    raw = np.empty(len(z))
    state = None
    with torch.no_grad():
        for start in range(0, len(z), 8192):
            h, state = net(
                torch.from_numpy(features[start : start + 8192])[None], state
            )
            raw[start : start + len(h[0])] = (
                h[0].numpy() @ np.asarray(spec["head"]) + spec["bias"]
            )
    rate = raw * spec["rate_scale"] + (base if spec["residual"] else 0)
    if spec["gate"]:
        rate[base == 0] = 0
    rate = np.clip(rate, -spec["rate_limit"], spec["rate_limit"])
    x = np.empty(len(z))
    x[0] = initial
    lo, hi = r["bounds"]
    for k in range(1, len(z)):
        x[k] = min(hi, max(lo, x[k - 1] + rate[k] / 64))
    return z, x, angle_values(model["angle"], x)


def fast_score(model, flights):
    sse = sst = n = 0
    for f in flights:
        y = f.values[32:, 2]
        p = fast_rollout(model, f.values[:, 0], f.values[0, 2])[1][32:]
        if not np.isfinite(p).all():
            return float("-inf"), float("inf")
        sse += np.sum((y - p) ** 2)
        sst += np.sum((y - y.mean()) ** 2)
        n += len(y)
    return float(100 * (1 - np.sqrt(sse / sst))), float(np.sqrt(sse / n))
