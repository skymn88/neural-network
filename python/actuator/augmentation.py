"""Optional TRAIN-only synthetic augmentation; never calibrate on val/test."""

import json, time
import numpy as np
from .data import FS, pulse_flights
from .experiment import evaluate, serial, write_workbook
from .models import val_rmse


def calibrate_train(train):
    if not train:
        raise ValueError("TRAIN flights required")
    data = np.concatenate([f.values for f in train])
    lo, hi = np.quantile(data[:, 2], [0.0001, 0.9999])
    span = hi - lo
    if span <= 0:
        raise ValueError("TRAIN RAM range is zero")
    holding = []
    moving_z = []
    moving_v = []
    rates = []
    for f in train:
        x = f.values[:, 2]
        v = np.r_[0, np.diff(x) * FS]
        interior = (x > lo + 0.03 * span) & (x < hi - 0.03 * span)
        hold = interior & (abs(v) < max(0.001 * span, 0.05))
        moving = interior & (abs(v) > max(0.01 * span, 0.1))
        holding.extend(f.values[hold, 1])
        moving_z.extend(f.values[moving, 1])
        moving_v.extend(v[moving])
        rates.extend(abs(v[moving]))
    if len(holding) < 64 or len(rates) < 32:
        raise ValueError("TRAIN needs interior holds and motion for augmentation")
    q = np.quantile(holding, [0.05, 0.95])
    center = float(np.median(holding))
    coeff = np.linalg.lstsq(
        np.c_[data[:, 0], np.ones(len(data))], data[:, 1], rcond=None
    )[0]
    if abs(coeff[0]) < 1e-8:
        raise ValueError("Current-to-DDV relation unidentifiable in TRAIN")
    direction = float(np.sign(np.corrcoef(moving_z, moving_v)[0, 1]))
    if not np.isfinite(direction) or direction == 0:
        raise ValueError("TRAIN rate direction unidentifiable")
    excitation = float(np.quantile(abs(data[:, 1] - center), 0.995))
    return {
        "ram_bounds": [float(lo), float(hi)],
        "null_range": q.tolist(),
        "drift_amplitude": float(max(np.ptp(q) * 0.15, 0.001)),
        "deadband": float(max(np.ptp(q) * 0.1, 0.001)),
        "max_rate": float(np.quantile(rates, 0.98)),
        "rate_direction": direction,
        "valve_gain": float(coeff[0]),
        "valve_bias": float(coeff[1]),
        "park_ddv": center - direction * excitation,
        "excitation": excitation,
        "angle_weights": np.polynomial.polynomial.polyfit(
            data[:, 2], data[:, 3], 3
        ).tolist(),
        "duration": float(
            max(900, min(5000, np.median([len(f.time) / FS for f in train])))
        ),
        "calibration_scope": "TRAIN measurements only; simplified synthetic family, not identified physics",
    }


def augment_train(groups, count, seed=123):
    if count < 0 or int(count) != count:
        raise ValueError("syntheticTrainFlights must be a nonnegative integer")
    copied = {s: list(fs) for s, fs in groups.items()}
    if count == 0:
        return copied, [], None
    calibration = calibrate_train(groups["train"])
    flights, _ = pulse_flights(
        seed=seed,
        durations=tuple([calibration["duration"]] * count),
        profile="flightLike",
        calibration=calibration,
    )
    for f in flights:
        f.name = "SYNTH_TRAIN_" + f.name
    copied["train"] += flights
    return copied, flights, calibration


def compare_augmentation(flights, groups, count, **kwargs):
    """Run A/B at the same settings; original real validation/test are untouched."""
    from .experiment import run

    if kwargs.get("source") != "user":
        raise ValueError(
            "syntheticTrainFlights A/B requires user data; no synthetic validation disguised as real"
        )
    total_start = time.perf_counter()
    include_test = kwargs.pop("include_test", False)
    original, base_models, _ = run(flights, groups, include_test=False, **kwargs)
    augmented, added, calibration = augment_train(
        groups, count, kwargs.get("seed", 42) + 1000
    )
    result, aug_models, out = run(
        flights + added, augmented, include_test=False, **kwargs
    )
    models = {**base_models, **{name + "__SYNTH": m for name, m in aug_models.items()}}
    scores = {name: val_rmse(m, groups["validation"]) for name, m in models.items()}
    winner = min(scores, key=scores.get)  # BEFORE any real test evaluation
    per, summary = evaluate(models, groups, include_test=include_test)
    baseline = original["selected"]
    augwinner = result["selected"] + "__SYNTH"
    result.update(
        selected=winner,
        summary=summary,
        per_flight=per,
        validation_scores=scores,
        dataset_sha256=result["dataset_sha256"],
        calibration=calibration,
        notice="Real-data A/B; synthetic samples only in augmented TRAIN; table TRAIN metrics use real TRAIN only",
    )
    result["config"].update(
        syntheticTrainFlights=count,
        syntheticProfile="flightLike",
        include_test=include_test,
    )
    result["timings"] = {
        "baseline_total": original["timings"]["total"],
        "augmented_total": result["timings"]["total"],
        "total": time.perf_counter() - total_start,
    }
    result["diagnostics"].append(
        {
            "diagnostic": "SyntheticTrainAB",
            "baseline_model": baseline,
            "augmented_model": augwinner,
            "validation_rmse_without": scores[baseline],
            "validation_rmse_with": scores[augwinner],
            "validation_rmse_delta": scores[augwinner] - scores[baseline],
            "synthetic_train_flights": count,
            "test_used_for_selection": False,
        }
    )
    ab = result["diagnostics"][-1]
    for split in ("validation", "test"):
        a = next(
            (
                r
                for r in summary
                if r["model"] == baseline and r["stage"] == "RAM_Cascade"
            ),
            None,
        )
        b = next(
            (
                r
                for r in summary
                if r["model"] == augwinner and r["stage"] == "RAM_Cascade"
            ),
            None,
        )
        key = "fit_" + split
        if a and b and key in a and key in b:
            ab[split + "_fit_without"] = a[key]
            ab[split + "_fit_with"] = b[key]
            ab[split + "_fit_delta"] = (
                b[key] - a[key] if a[key] is not None and b[key] is not None else None
            )
    result["search"] = original["search"] + [
        dict(row, training_variant="augmented") for row in result["search"]
    ]
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    (out / "models.json").write_text(json.dumps(serial(models)))
    truth_arrays = {
        f"{k}_{i}": v
        for i, f in enumerate(added)
        if f.truth
        for k, v in f.truth.items()
        if isinstance(v, np.ndarray)
    }
    if truth_arrays:
        np.savez_compressed(out / "synthetic_truth.npz", **truth_arrays)
        (out / "synthetic_truth.json").write_text(
            json.dumps(
                serial(
                    [
                        {
                            k: v
                            for k, v in f.truth.items()
                            if not isinstance(v, np.ndarray)
                        }
                        for f in added
                        if f.truth
                    ]
                ),
                indent=2,
            )
        )
    write_workbook(out, result)
    return serial(result), models, out
