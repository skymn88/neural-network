"""Portable batch inference: NumPy only; no training packages or measured feedback."""

import numpy as np

FS = 64


def lags(u, length):
    padded = np.pad(np.asarray(u), (length, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, length + 1)[:, ::-1].copy()


def ddv_features(current, scale, length, basis="legacy"):
    z = lags(current / scale, length)
    if basis == "rich":
        return np.column_stack(
            [np.ones(len(z)), z, np.abs(z)]
            + [np.tanh(z / s) for s in [0.5, 1, 2, 4, 8]]
            + [np.sign(z) * np.maximum(np.abs(z) - d, 0) for d in [0.05, 0.15]]
        )
    return np.column_stack([np.ones(len(z)), z, np.tanh(z), np.tanh(z / 2), np.abs(z)])


def predict_ddv(model, current):
    return ddv_features(
        current, model["scale"], model["lags"], model.get("basis", "legacy")
    ) @ np.asarray(model["weights"])


def ram_features(u, scale, length):
    z = lags(u / scale, length)
    return np.column_stack([np.ones(len(z)), z, np.tanh(z), np.abs(z)])


def angle_values(model, x):
    return np.polynomial.polynomial.polyval(x / model["scale"], model["weights"])


def rollout(model, current, x0, ddv_override=None):
    ddv = (
        predict_ddv(model["ddv"], current)
        if ddv_override is None
        else np.asarray(ddv_override)
    )
    m = model["ram"]
    n = len(current)
    if m["kind"] == "null_rate":
        u, _ = neutral_features(ddv, m["observer"])
        delay = m["delay"]
        if delay:
            u = np.r_[np.repeat(u[0], delay), u[: len(u) - delay]]
        rate = np.interp(u, m["knots"], m["values"])
        if "map_rate_limit" in m:
            rate = np.clip(rate, -m["map_rate_limit"], m["map_rate_limit"])
        if "zero_band" in m:
            rate[(u >= m["zero_band"][0]) & (u <= m["zero_band"][1])] = 0.0
        x = np.empty(n)
        x[0] = x0
        lo, hi = m["bounds"]
        for k in range(1, n):
            x[k] = min(hi, max(lo, x[k - 1] + rate[k] / FS))
    elif m["kind"] == "sequence_rate":
        from .sequence_inference import sequence_features, sequence_rate

        features, base_rate = sequence_features(ddv, m["reference"])
        rate = sequence_rate(m["sequence"], features, base_rate)
        x = np.empty(n)
        x[0] = x0
        lo, hi = m["bounds"]
        for k in range(1, n):
            x[k] = min(hi, max(lo, x[k - 1] + rate[k] / FS))
    elif m["kind"] == "rate_ensemble":
        from .sequence_inference import ensemble_rate, integrate_rate

        x = integrate_rate(ensemble_rate(m, ddv), x0, m["bounds"])
    elif m["kind"] == "seed_ensemble":
        from .seed_ensemble import integrate_members

        inputs = (
            [predict_ddv(d, current) for d in m["ddv_members"]]
            if ddv_override is None
            else [ddv] * len(m["members"])
        )
        ddv = np.mean(inputs, axis=0)
        x = integrate_members(m["members"], inputs, x0)
    elif m["kind"] == "oe":
        drive = ram_features(ddv, m["scale"], m["lags"]) @ np.asarray(m["weights"])
        drive[0] = 0
        a = 1 + m["beta"] / FS
        x = np.empty(n)
        x[0] = x0
        for k in range(1, n):
            x[k] = a * x[k - 1] + drive[k] / FS
    elif m["kind"] == "monotone":
        x = np.empty(n)
        x[0] = x0
        v = 0.0
        for k in range(1, n):
            q = (
                mono_numpy(m["g1"], ddv[k] / m["ddv_scale"] * m["direction"])
                - m["leak"] * (x[k - 1] - m["x_mean"]) / m["x_scale"]
                + m["offset"]
            )
            target = mono_numpy(m["g2"], q)
            v = (m["memory"] * v / m["v_scale"] + (1 - m["memory"]) * target) * m[
                "v_scale"
            ]
            x[k] = x[k - 1] + v / FS
            if not np.isfinite(x[k]) or abs(x[k] - m["x_mean"]) > 1e6 * m["x_scale"]:
                x[k:] = np.nan
                break
    else:
        W1, W2 = np.asarray(m["W1"]), np.asarray(m["W2"])
        b1, b2 = np.asarray(m["b1"]), float(m["b2"])
        x = np.empty(n)
        x[0] = x0
        v = 0.0
        for k in range(1, n):
            z = np.array(
                [
                    ddv[k] / m["ddv_scale"],
                    (x[k - 1] - m["x_mean"]) / m["x_scale"],
                    v / m["v_scale"],
                ]
            )
            v = (v / m["v_scale"] + float(W2 @ np.tanh(W1 @ z + b1)) + b2) * m[
                "v_scale"
            ]
            x[k] = x[k - 1] + v / FS
            if not np.isfinite(x[k]) or abs(x[k] - m["x_mean"]) > 1e6 * m["x_scale"]:
                x[k:] = np.nan
                break
    return ddv, x, angle_values(model["angle"], x)


def mono_numpy(g, z):
    return g["linear"] * z + float(
        np.asarray(g["w2"]) @ np.tanh(np.asarray(g["w1"]) * z + np.asarray(g["b"]))
    )


def neutral_features(ddv, spec):
    center = np.empty(len(ddv))
    estimate = spec["center"]
    ready = False
    alpha = 1 - np.exp(-1 / (FS * spec["tau"]))
    previous = ddv[0]
    mode = spec.get("mode", "legacy")
    was_quiet = False
    quiet = np.zeros(len(ddv), dtype=bool)
    if mode != "legacy":
        quiet = (
            np.ptp(lags(ddv, spec["quiet_window"] - 1), axis=1) <= spec["quiet_range"]
        )
    high = low = None
    high_at = low_at = -1000000
    previous_direction = 0
    for k, z in enumerate(ddv):
        if mode != "legacy" and spec["adaptive"]:
            is_quiet = bool(quiet[k] and spec["gate"][0] <= z <= spec["gate"][1])
            if is_quiet:
                if not was_quiet:
                    estimate = z
                    ready = True
                else:
                    estimate += alpha * (z - estimate)
            elif mode == "envelope":
                direction = np.sign(z - previous)
                width = spec["gate"][1] - spec["gate"][0]
                if (
                    previous_direction > 0
                    and direction < 0
                    and previous > spec["gate"][1] + width * 0.5
                ):
                    high = previous
                    high_at = k
                if (
                    previous_direction < 0
                    and direction > 0
                    and previous < spec["gate"][0] - width * 0.5
                ):
                    low = previous
                    low_at = k
                if (
                    high is not None
                    and low is not None
                    and k - min(high_at, low_at) < spec["envelope_age"]
                ):
                    midpoint = 0.5 * (high + low)
                    if spec["gate"][0] <= midpoint <= spec["gate"][1]:
                        estimate += 0.05 * (midpoint - estimate)
                if direction:
                    previous_direction = direction
            center[k] = estimate
            previous = z
            was_quiet = is_quiet
            continue
        if (
            spec["adaptive"]
            and spec["gate"][0] <= z <= spec["gate"][1]
            and abs(z - previous) < spec["max_step"]
        ):
            if not ready:
                estimate = z
                ready = True
            else:
                estimate += alpha * (z - estimate)
        center[k] = estimate
        previous = z
    return ddv - center, center
