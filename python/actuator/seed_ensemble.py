"""Equal physical velocity averaging before shared position integration (NumPy)."""

import copy
import json
import numpy as np
from .inference import FS, neutral_features, ram_features, mono_numpy
from .sequence_inference import sequence_features, sequence_rate


def make_ensemble(models, seeds):
    if len(models) != len(seeds) or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Distinct matched seeds required")
    if all(m["ram"]["kind"] == "sequence_rate" for m in models):
        from .generalization import make_rate_ensemble

        return make_rate_ensemble(models, seeds)
    first = models[0]
    for model in models:
        for key in ("angle",):
            if json.dumps(model[key], sort_keys=True) != json.dumps(
                first[key], sort_keys=True
            ):
                raise ValueError("Ensemble requires identical angle models")
    return dict(
        ddv=copy.deepcopy(first["ddv"]),
        angle=copy.deepcopy(first["angle"]),
        ram=dict(
            kind="seed_ensemble",
            members=[copy.deepcopy(m["ram"]) for m in models],
            ddv_members=[copy.deepcopy(m["ddv"]) for m in models],
        ),
        training_seeds=list(seeds),
        ensemble_policy="equal physical velocities; shared RAM state",
    )


def integrate_members(members, ddvs, x0):
    drives = []
    for m, ddv in zip(members, ddvs):
        kind = m["kind"]
        if kind == "sequence_rate":
            features, base = sequence_features(ddv, m["reference"])
            drive = sequence_rate(m["sequence"], features, base)
        elif kind == "null_rate":
            u, _ = neutral_features(ddv, m["observer"])
            delay = m["delay"]
            if delay:
                u = np.r_[np.repeat(u[0], delay), u[:-delay]]
            drive = np.interp(u, m["knots"], m["values"])
            if "map_rate_limit" in m:
                drive = np.clip(drive, -m["map_rate_limit"], m["map_rate_limit"])
            if "zero_band" in m:
                drive[(u >= m["zero_band"][0]) & (u <= m["zero_band"][1])] = 0
        elif kind == "oe":
            drive = ram_features(ddv, m["scale"], m["lags"]) @ np.asarray(m["weights"])
        else:
            drive = None
        drives.append(drive)
    bounds = [m.get("bounds") for m in members]
    if any(b != bounds[0] for b in bounds):
        raise ValueError("Members must share bounds")
    x = np.empty(len(ddvs[0]))
    x[0] = x0
    velocities = np.zeros(len(members))
    for k in range(1, len(x)):
        for j, (m, drive) in enumerate(zip(members, drives)):
            kind = m["kind"]
            if drive is not None:
                velocities[j] = drive[k] + (m["beta"] * x[k - 1] if kind == "oe" else 0)
            elif kind == "monotone":
                q = (
                    mono_numpy(m["g1"], ddvs[j][k] / m["ddv_scale"] * m["direction"])
                    - m["leak"] * (x[k - 1] - m["x_mean"]) / m["x_scale"]
                    + m["offset"]
                )
                target = mono_numpy(m["g2"], q)
                velocities[j] = (
                    m["memory"] * velocities[j]
                    + (1 - m["memory"]) * target * m["v_scale"]
                )
            else:
                z = np.array(
                    [
                        ddvs[j][k] / m["ddv_scale"],
                        (x[k - 1] - m["x_mean"]) / m["x_scale"],
                        velocities[j] / m["v_scale"],
                    ]
                )
                velocities[j] += (
                    float(
                        np.asarray(m["W2"]) @ np.tanh(np.asarray(m["W1"]) @ z + m["b1"])
                    )
                    + m["b2"]
                ) * m["v_scale"]
        x[k] = x[k - 1] + np.mean(velocities) / FS
        if bounds[0] is not None:
            x[k] = np.clip(x[k], *bounds[0])
        if not np.isfinite(x[k]) or abs(x[k]) > 1e12:
            x[k:] = np.nan
            break
    return x
