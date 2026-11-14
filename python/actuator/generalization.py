"""Frozen mixed-valve cohorts and TRAIN-only helpers for a controlled scaling audit."""

import copy
import json
import numpy as np
from .data import FS, FLIGHT_LIKE_DURATIONS, pulse_flights, fingerprint
from .models import fit_ddv
from .null_rate import fit_null_rate, refine_window_rate

PROTOCOL = dict(
    original_seed=621,
    validation_seed=910247,
    extra_train_seed=741203,
    validation_durations=[2100, 2400, 2600, 2200, 2800, 2500],
    extra_train_durations=[FLIGHT_LIKE_DURATIONS[i % 11] for i in range(14)],
    variation=dict(gain_sd=0.03, bias_sd=0.025, drift_amp=0.02),
    training_seeds=[42, 43, 44],
    train_counts=[7, 14, 21],
)


def generalization_data():
    old, split = pulse_flights(
        seed=PROTOCOL["original_seed"],
        profile="flightLike",
        valve_variation=PROTOCOL["variation"],
    )
    added, _ = pulse_flights(
        seed=PROTOCOL["extra_train_seed"],
        durations=PROTOCOL["extra_train_durations"],
        profile="flightLike",
        valve_variation=PROTOCOL["variation"],
    )
    validation, _ = pulse_flights(
        seed=PROTOCOL["validation_seed"],
        durations=PROTOCOL["validation_durations"],
        profile="flightLike",
        valve_variation=PROTOCOL["variation"],
    )
    for i, f in enumerate(added):
        f.name = f"TRAIN_EXTRA_{i + 1:02}"
    for i, f in enumerate(validation):
        f.name = f"VAL_NEW_{i + 1:02}"
    return old, split, added, validation


def training_speed_limit(train):
    speeds = np.concatenate([abs(np.diff(f.values[:, 2]) * FS) for f in train])
    limit = float(np.quantile(speeds, 0.999))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("TRAIN speed quantile must be positive and finite")
    return limit


def shared_reference(train, previous):
    # Upstream hyperparameters frozen from previous model; no new validation in fitting.
    old = previous["ddv"]
    ddv, _ = fit_ddv(
        train, train, length=old["lags"], basis=old["basis"], fixed_ridge=old["ridge"]
    )
    values = np.concatenate([f.values for f in train])
    scale = max(float(np.std(values[:, 2])), 1e-9)
    degree = len(previous["angle"]["weights"]) - 1
    angle = dict(
        scale=scale,
        weights=np.polynomial.polynomial.polyfit(
            values[:, 2] / scale, values[:, 3], degree
        ).tolist(),
    )
    ref = fit_null_rate(
        train,
        ddv,
        angle,
        adaptive=True,
        tau=2,
        knots=81,
        ridge=0.001,
        observer="stationary",
        gate_source="predicted",
        gate_expansion=1.5,
    )
    return refine_window_rate(ref, train, seconds=2, ridge=0.01)


def clipped_reference(reference, limit):
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Positive finite TRAIN limit required")
    ref = copy.deepcopy(reference)
    ref["ram"]["map_rate_limit"] = float(limit)
    return ref


def make_rate_ensemble(models, seeds):
    if len(models) < 2 or len(seeds) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("At least two matched seed models required")
    first = models[0]
    for m in models:
        if m["ram"]["kind"] != "sequence_rate":
            raise ValueError("Sequence members required")
        if (
            any(
                json.dumps(m[k], sort_keys=True) != json.dumps(first[k], sort_keys=True)
                for k in ("ddv", "angle")
            )
            or m["ram"]["bounds"] != first["ram"]["bounds"]
        ):
            raise ValueError("Ensemble must share DDV, angle and TRAIN bounds")
    return dict(
        ddv=copy.deepcopy(first["ddv"]),
        angle=copy.deepcopy(first["angle"]),
        ram=dict(
            kind="rate_ensemble",
            members=[copy.deepcopy(m["ram"]) for m in models],
            bounds=first["ram"]["bounds"],
        ),
        training_seeds=list(seeds),
        ensemble_policy="equal rates before one RAM integration",
    )


def cohort_manifest(old, split, added, validation):
    return dict(
        protocol=copy.deepcopy(PROTOCOL),
        original_sha256=fingerprint(old),
        extra_train_sha256=fingerprint(added),
        extra_validation_sha256=fingerprint(validation),
        original_split={s: [f.name for f in fs] for s, fs in split.items()},
        added_train=[f.name for f in added],
        added_validation=[f.name for f in validation],
    )
