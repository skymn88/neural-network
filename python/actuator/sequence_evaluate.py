"""Optional Torch-accelerated scoring, same exported model and full-flight state."""

import numpy as np
from .inference import rollout, predict_ddv, angle_values
from .sequence_inference import (
    sequence_features,
    sequence_raw,
    combine_rate,
    integrate_rate,
)


def recurrent_states(spec, features):
    """Full causal hidden sequence, carrying state across bounded Torch chunks."""
    import torch

    torch.set_num_threads(1)
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
    features = (features - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
    states = np.empty((len(features), spec["hidden"]))
    state = None
    with torch.no_grad():
        for start in range(0, len(features), 8192):
            h, state = net(
                torch.from_numpy(features[start : start + 8192])[None], state
            )
            states[start : start + len(h[0])] = h[0].numpy()
    return states


def fast_rate(ram, ddv):
    if ram["kind"] == "rate_ensemble":
        return np.mean([fast_rate(member, ddv) for member in ram["members"]], axis=0)
    spec = ram["sequence"]
    features, base = sequence_features(ddv, ram["reference"])
    if spec["family"] in ("gru", "lstm", "elman"):
        raw = recurrent_states(spec, features) @ np.asarray(spec["head"]) + spec["bias"]
    else:
        raw = sequence_raw(spec, features)
    return combine_rate(spec, raw, base)


def fast_rollout(model, current, initial, ddv_override=None):
    ram = model["ram"]
    if ram["kind"] not in ("sequence_rate", "rate_ensemble"):
        return rollout(model, current, initial, ddv_override)
    z = (
        predict_ddv(model["ddv"], current)
        if ddv_override is None
        else np.asarray(ddv_override)
    )
    x = integrate_rate(fast_rate(ram, z), initial, ram["bounds"])
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
