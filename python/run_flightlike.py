"""At most ten candidate experiments; stop at validation RAM Cascade Fit >=95.
No TEST evaluation until the winner and the stopping decision are frozen.
"""

import argparse, json, time, platform
from datetime import datetime, timezone
import numpy as np
from actuator.data import ROOT, FS, pulse_flights, FLIGHT_LIKE_DURATIONS, fingerprint
from actuator.models import fit_ddv, fit_angle, fit_neural
from actuator.null_rate import fit_null_rate
from actuator.inference import rollout, neutral_features
from actuator.experiment import evaluate, serial, write_workbook


def score(model, flights):
    sse = sst = n = 0
    for f in flights:
        y = f.values[32:, 2]
        p = rollout(model, f.values[:, 0], f.values[0, 2])[1][32:]
        if not np.all(np.isfinite(p)):
            return float("-inf"), float("inf")
        sse += np.sum((y - p) ** 2)
        sst += np.sum((y - y.mean()) ** 2)
        n += len(y)
    return float(100 * (1 - np.sqrt(sse / sst))), float(np.sqrt(sse / n))


def run(limit=10):
    start = time.perf_counter()
    flights, groups = pulse_flights(
        seed=621, durations=FLIGHT_LIKE_DURATIONS, profile="flightLike"
    )
    print(
        "Generated 11 flightLike flights; TRAIN/VAL/TEST 7/2/2; generator frozen",
        flush=True,
    )
    folder = (
        ROOT
        / "python/results"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "flights.npz",
        **{f"f{i}": np.c_[f.time, f.values] for i, f in enumerate(flights)},
    )
    np.savez_compressed(
        folder / "truth.npz",
        **{
            f"{key}_{i}": f.truth[key]
            for i, f in enumerate(flights)
            for key in ("null", "ram_clean", "ddv_clean", "rate_requested", "phase")
        },
    )
    (folder / "truth.json").write_text(
        json.dumps(
            serial(
                [
                    {
                        k: v
                        for k, v in f.truth.items()
                        if k
                        not in (
                            "null",
                            "ram_clean",
                            "ddv_clean",
                            "rate_requested",
                            "phase",
                        )
                    }
                    for f in flights
                ]
            ),
            indent=2,
        )
    )
    (folder / "flights.json").write_text(json.dumps([f.name for f in flights]))
    t = time.perf_counter()
    ddv, search = fit_ddv(groups["train"], groups["validation"])
    angle = fit_angle(groups["train"], groups["validation"])
    timings = {"ddv_angle": time.perf_counter() - t}
    models = {}
    trials = []
    settings = [
        ("NN_joint", {}),
        ("Rate_static", {"adaptive": False}),
        ("Rate_adaptive", {"adaptive": True}),
        ("Rate_fast_null", {"adaptive": True, "tau": 0.15}),
        ("Rate_dense", {"adaptive": True, "tau": 0.15, "knots": 161}),
        ("Rate_delayed", {"adaptive": True, "tau": 0.15, "knots": 161, "delay": 2}),
        ("Rate_smooth", {"adaptive": True, "tau": 0.15, "knots": 161, "ridge": 1e-3}),
        ("Rate_slow_null", {"adaptive": True, "tau": 2, "knots": 161}),
        ("Rate_no_delay", {"adaptive": True, "tau": 0.15, "knots": 161, "delay": 0}),
        ("Rate_coarse", {"adaptive": True, "tau": 0.15, "knots": 41}),
    ]
    for index, (name, cfg) in enumerate(settings[:limit], 1):
        t = time.perf_counter()
        if name == "NN_joint":
            mm, rows = fit_neural(
                groups["train"],
                groups["validation"],
                ddv,
                angle,
                steps=160,
                horizon=128,
                hidden=16,
                warm_steps=100,
            )
            m = mm["NN_joint"]
            search += rows
        else:
            m = fit_null_rate(groups["train"], ddv, angle, **cfg)
        models[name] = m
        fit, rmse = score(m, groups["validation"])
        elapsed = time.perf_counter() - t
        trials.append(
            {
                "iteration": index,
                "model": name,
                "validation_fit": fit,
                "validation_rmse": rmse,
                "seconds": elapsed,
                **cfg,
            }
        )
        timings[name] = elapsed
        print(
            f"ITERATION {index}/{limit}: {name} VAL Fit={fit:.4f} RMSE={rmse:.5g} time={elapsed:.1f}s",
            flush=True,
        )
        (folder / "search_progress.json").write_text(
            json.dumps(serial(trials), indent=2)
        )
        (folder / "models.json").write_text(json.dumps(serial(models)))
        if fit >= 95:
            break
    winner = min(trials, key=lambda r: r["validation_rmse"])["model"]
    print("FROZEN winner:", winner, "TEST evaluated only now", flush=True)
    t = time.perf_counter()
    per, summary = evaluate(models, groups, include_test=True)
    timings["evaluation"] = time.perf_counter() - t
    diagnostics = []
    for split, fs in groups.items():
        for f in fs:
            z, x, _ = rollout(models[winner], f.values[:, 0], f.values[0, 2])
            observed = f.values[:, 2]
            v = np.r_[0, np.diff(observed) * FS]
            # Measurement-based motion/hold labels, no Truth used for scoring.
            moving = abs(v) > 1
            holding = ~moving
            err = x - observed
            diagnostics.append(
                {
                    "diagnostic": "MotionHold",
                    "flight": f.name,
                    "split": split,
                    "motion_rmse": float(np.sqrt(np.mean(err[moving] ** 2)))
                    if moving.any()
                    else None,
                    "hold_rmse": float(np.sqrt(np.mean(err[holding] ** 2)))
                    if holding.any()
                    else None,
                    "measured_hold_fraction": float(np.mean(holding)),
                    "hold_rate_rmse": float(
                        np.sqrt(np.mean((np.r_[0, np.diff(x) * FS][holding]) ** 2))
                    )
                    if holding.any()
                    else None,
                }
            )
            m = models[winner]["ram"]
            if m["kind"] == "null_rate":
                _, center = neutral_features(z, m["observer"])
                gate = (z >= m["observer"]["gate"][0]) & (z <= m["observer"]["gate"][1])
                diagnostics.append(
                    {
                        "diagnostic": "StaticNullTruth",
                        "flight": f.name,
                        "split": split,
                        "true_null_mean": float(np.mean(f.truth["null"])),
                        "estimated_null_mean": float(np.mean(center)),
                        "hold_null_rmse": float(
                            np.sqrt(
                                np.mean((center[gate] - f.truth["null"][gate]) ** 2)
                            )
                        )
                        if gate.any()
                        else None,
                        "true_deadband": f.truth["deadband"],
                    }
                )
    timings["total"] = time.perf_counter() - start
    result = {
        "id": folder.name,
        "selected": winner,
        "config": {
            "source": "flightLike",
            "syntheticProfile": "flightLike",
            "seed": 621,
            "fs": 64,
            "include_test": True,
            "max_iterations": limit,
            "target_validation_fit": 95,
            "syntheticTrainFlights": 0,
        },
        "summary": summary,
        "per_flight": per,
        "search": search + [dict(r, stage="RAM") for r in trials],
        "diagnostics": diagnostics,
        "split": [
            {"flight": f.name, "split": s, "samples": len(f.time)}
            for s, fs in groups.items()
            for f in fs
        ],
        "dataset_sha256": fingerprint(flights),
        "validation_scores": {r["model"]: r["validation_rmse"] for r in trials},
        "timings": timings,
        "notice": "Synthetic flightLike software experiment; real-flight augmentation A/B NOT performed",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "trials": trials,
        "selection_policy": "At most 10 predeclared candidates; stop if pooled full-flight validation RAM Cascade Fit>=95; test read after freezing",
        "generator_assumptions": "Photo-informed only. Null drift is visible in neutral holding current; arbitrary independent hidden offset is not identifiable from current alone.",
    }
    (folder / "result.json").write_text(json.dumps(serial(result), indent=2))
    write_workbook(folder, result)
    print("RESULT", folder, flush=True)
    for r in summary:
        if r["model"] == winner:
            print(r, flush=True)
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.limit <= 10:
        p.error("limit must be 1..10")
    run(args.limit)
