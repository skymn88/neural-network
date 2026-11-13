"""TRAIN-only clipped trajectory Gauss-Newton readout refinement; frozen memory."""

import copy
import numpy as np
from .inference import predict_ddv
from .sequence_inference import sequence_features
from .sequence_evaluate import recurrent_states


def clipped_trajectory(design, offset, weights, initial, bounds, rate_limit):
    raw = offset + design @ weights
    rates = np.clip(raw, -rate_limit, rate_limit)
    x = np.empty(len(raw))
    x[0] = initial
    reset = np.zeros(len(raw), dtype=bool)
    reset[0] = True
    lo, hi = bounds
    for k in range(1, len(raw)):
        candidate = x[k - 1] + rates[k] / 64
        x[k] = min(hi, max(lo, candidate))
        reset[k] = candidate <= lo or candidate >= hi
    # Chain derivative resets whenever position clips; saturated rates have zero derivative.
    derivative = design.copy()
    derivative[abs(raw) >= rate_limit] = 0
    derivative[0] = 0
    cumulative = np.cumsum(derivative / 64, axis=0)
    last = np.maximum.accumulate(np.where(reset, np.arange(len(raw)), 0))
    jac = cumulative - cumulative[last]
    return x, jac


def refine_sequence_head(model, train, iterations=3, ridge=0.01, focus=1.0):
    if (
        not isinstance(iterations, int)
        or iterations < 1
        or not np.isfinite([ridge, focus]).all()
        or ridge <= 0
        or focus < 0
    ):
        raise ValueError("Positive iterations/ridge and nonnegative focus required")
    m = copy.deepcopy(model)
    r = m["ram"]
    spec = r["sequence"]
    if spec["family"] not in ("gru", "lstm", "elman"):
        raise ValueError("Head refinement requires recurrent sequence model")
    weights = np.r_[spec["head"], spec["bias"]].astype(float)
    cached = []
    for f in train:
        z = predict_ddv(m["ddv"], f.values[:, 0])
        features, base = sequence_features(z, r["reference"])
        design = (
            np.column_stack([recurrent_states(spec, features), np.ones(len(z))])
            * spec["rate_scale"]
        )
        offset = base.copy() if spec["residual"] else np.zeros(len(z))
        if spec["gate"]:
            design[base == 0] = 0
            offset[base == 0] = 0
        y = f.values[:, 2]
        sst = max(np.sum((y[32:] - y[32:].mean()) ** 2), 1e-12)
        cached.append((design, offset, y, sst))

    def assess(w, jacobian=False):
        ratios = []
        pairs = []
        for A, b, y, sst in cached:
            x, J = clipped_trajectory(A, b, w, y[0], r["bounds"], spec["rate_limit"])
            ratios.append(np.sum((y[32:] - x[32:]) ** 2) / sst)
            if jacobian:
                pairs.append((J[32:], y[32:] - x[32:], sst))
        return np.asarray(ratios), pairs

    trace = []
    for iteration in range(iterations):
        ratios, pairs = assess(weights, True)
        importance = np.maximum(ratios, 1e-8) ** focus
        importance /= importance.sum()
        G = np.zeros((len(weights), len(weights)))
        b = np.zeros(len(weights))
        for importance_i, (J, error, sst) in zip(importance, pairs):
            G += importance_i * (J.T @ J) / sst
            b += importance_i * (J.T @ error) / sst
        scale = np.sqrt(np.maximum(np.diag(G), 1e-10))
        step = (
            np.linalg.solve(
                G / np.outer(scale, scale) + ridge * np.eye(len(weights)), b / scale
            )
            / scale
        )
        best = float(ratios.max())
        best_w = weights
        # All backtracking decisions use TRAIN worst normalized squared error only.
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = weights + fraction * step
            trial, _ = assess(candidate)
            if trial.max() < best:
                best = float(trial.max())
                best_w = candidate
        trace.append(
            dict(iteration=iteration + 1, train_worst_fit=100 * (1 - np.sqrt(best)))
        )
        if np.array_equal(best_w, weights):
            break
        weights = best_w
    spec["head"] = weights[:-1].tolist()
    spec["bias"] = float(weights[-1])
    spec["head_refinement"] = dict(
        iterations=iterations, ridge=ridge, focus=focus, trace=trace
    )
    return m
