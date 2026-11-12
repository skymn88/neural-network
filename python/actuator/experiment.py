import json, time, platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import ROOT, FS, fingerprint
from .models import (
    fit_ddv,
    fit_angle,
    fit_oe,
    fit_neural,
    fit_stable,
    rollout,
    val_rmse,
    angle_values,
)


def metrics(y, p):
    e = p - y
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(e**2))
    return {
        "fit": 100 * (1 - np.sqrt(sse / sst)) if sst > 1e-18 else None,
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "observed_std": float(np.std(y)),
        "sse": sse,
        "sst": sst,
        "samples": len(y),
        "coverage": float(np.mean(np.isfinite(p)) * 100),
    }


def evaluate(models, groups, include_test=True):
    per = []
    for method, model in models.items():
        for split, flights in groups.items():
            if split == "test" and not include_test:
                continue
            for f in flights:
                ddv, x, a = rollout(model, f.values[:, 0], f.values[0, 2])
                component = rollout(
                    model, f.values[:, 0], f.values[0, 2], f.values[:, 1]
                )[1]
                for stage, y, p in [
                    ("DDV", f.values[:, 1], ddv),
                    ("RAM_Component", f.values[:, 2], component),
                    ("RAM_Cascade", f.values[:, 2], x),
                    (
                        "Angle_Component",
                        f.values[:, 3],
                        angle_values(model["angle"], f.values[:, 2]),
                    ),
                    ("Angle_Cascade", f.values[:, 3], a),
                ]:
                    per.append(
                        dict(
                            model=method,
                            stage=stage,
                            split=split,
                            flight=f.name,
                            **metrics(y[32:], p[32:]),
                        )
                    )
    summary = []
    for method in models:
        for stage in [
            "DDV",
            "RAM_Component",
            "RAM_Cascade",
            "Angle_Component",
            "Angle_Cascade",
        ]:
            row = {"model": method, "stage": stage}
            for split in ("train", "validation", "test"):
                z = [
                    r
                    for r in per
                    if r["model"] == method
                    and r["stage"] == stage
                    and r["split"] == split
                ]
                if z:
                    sse = sum(r["sse"] for r in z)
                    sst = sum(r["sst"] for r in z)
                    n = sum(r["samples"] for r in z)
                    row["fit_" + split] = (
                        100 * (1 - np.sqrt(sse / sst)) if sst > 1e-18 else None
                    )
                    row["rmse_" + split] = np.sqrt(sse / n)
            summary.append(row)
    return per, summary


def diagnostics(models, train):
    scale = np.std(np.concatenate([f.values[:, 0] for f in train]))
    u = np.r_[np.ones(128), np.zeros(320), -np.ones(256), np.zeros(320)] * scale
    x0 = float(np.median(np.concatenate([f.values[:, 2] for f in train])))
    rows = []
    for name, m in models.items():
        z, x, a = rollout(m, u, x0)
        for stage, y in [("RAM", x), ("Angle", a)]:
            for start, end in [(128, 448), (704, 1024)]:
                slope = np.polyfit(np.arange(end - start) / FS, y[start:end], 1)[0]
                rows.append(
                    {
                        "model": name,
                        "diagnostic": "Synthetic PulseHold",
                        "stage": stage,
                        "start_seconds": start / FS,
                        "hold_slope": slope,
                        "initial_level": y[start - 1],
                        "final_level": y[end - 1],
                    }
                )
    return rows


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(
    flights,
    groups,
    source="bundled",
    steps=160,
    horizon=64,
    hidden=16,
    seed=42,
    methods=("oe", "neural"),
    include_test=False,
    log=print,
    ddv_basis="rich",
    warm_steps=100,
):
    if not methods or set(methods) - {"oe", "neural", "stable"}:
        raise ValueError("Choose oe, neural and/or stable")
    start = time.perf_counter()
    search = []
    timings = {}
    models = {}
    t = time.perf_counter()
    ddv, rows = fit_ddv(groups["train"], groups["validation"], basis=ddv_basis)
    angle = fit_angle(groups["train"], groups["validation"])
    search += rows
    timings["ddv_angle"] = time.perf_counter() - t
    log("DDV and angle fitted; starting RAM candidates")
    if "oe" in methods:
        t = time.perf_counter()
        models["OE"], rows = fit_oe(groups["train"], groups["validation"], ddv, angle)
        search += rows
        timings["OE"] = time.perf_counter() - t
        log(
            f"OE validation RAM RMSE={val_rmse(models['OE'], groups['validation']):.6g}"
        )
    if "neural" in methods:
        t = time.perf_counter()
        nn, rows = fit_neural(
            groups["train"],
            groups["validation"],
            ddv,
            angle,
            steps,
            horizon,
            seed,
            hidden,
            log,
            warm_steps,
        )
        models.update(nn)
        search += rows
        timings["neural"] = time.perf_counter() - t
    if "stable" in methods:
        t = time.perf_counter()
        models["NN_stable"], rows = fit_stable(
            groups["train"],
            groups["validation"],
            ddv,
            angle,
            steps,
            horizon,
            seed,
            hidden,
            log,
        )
        search += rows
        timings["NN_stable"] = time.perf_counter() - t
    scores = {name: val_rmse(m, groups["validation"]) for name, m in models.items()}
    selected = min(scores, key=scores.get)
    if not np.isfinite(scores[selected]):
        raise ValueError("No finite validation candidate")
    log(f"Frozen selection: {selected}; test used for selection: NO")
    t = time.perf_counter()
    per, summary = evaluate(models, groups, include_test)
    diag = diagnostics(models, groups["train"])
    timings["evaluation"] = time.perf_counter() - t
    config = dict(
        source=source,
        steps=steps,
        horizon=horizon,
        hidden=hidden,
        seed=seed,
        methods=list(methods),
        fs=FS,
        include_test=include_test,
        ddv_basis=ddv_basis,
        warm_steps=warm_steps,
    )
    manifest = [
        {"flight": f.name, "split": s, "samples": len(f.time)}
        for s, fs in groups.items()
        for f in fs
    ]
    result = {
        "selected": selected,
        "config": config,
        "summary": summary,
        "per_flight": per,
        "search": search,
        "diagnostics": diag,
        "split": manifest,
        "dataset_sha256": fingerprint(flights),
        "timings": timings,
        "validation_scores": scores,
        "notice": "Synthetic software experiment, not real-flight validation"
        if source != "user"
        else "User data; selection uses validation only",
        "versions": {
            "python": platform.python_version(),
            **{
                k: importlib.metadata.version(k)
                for k in ("numpy", "scipy", "torch", "pandas", "openpyxl")
            },
        },
        "algorithm_revision": "py-0.1",
        "selection_policy": "minimum full-flight validation RAM_Cascade RMSE; test excluded",
    }
    t = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "python/results" / stamp
    out.mkdir(parents=True)
    result["id"] = stamp
    (out / "models.json").write_text(json.dumps(serial(models)), encoding="utf8")
    np.savez_compressed(
        out / "flights.npz",
        **{f"f{i}": np.column_stack([f.time, f.values]) for i, f in enumerate(flights)},
    )
    (out / "flights.json").write_text(
        json.dumps([f.name for f in flights]), encoding="utf8"
    )
    # First write is timed; a final rewrite records that duration and total in XLSX.
    write_workbook(out, result)
    timings["report_write"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - start
    result["timing_note"] = (
        "Total includes training, evaluation and initial artifact write; excludes final timing metadata rewrite."
    )
    (out / "result.json").write_text(
        json.dumps(serial(result), indent=2), encoding="utf8"
    )
    write_workbook(out, result)
    log(f"Finished: {out}")
    return serial(result), models, out


def write_workbook(out, result):
    """Exactly five sheets; settings/timings accompany the split manifest."""
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as writer:
        for sheet, key in [
            ("Summary", "summary"),
            ("PerFlight", "per_flight"),
            ("Search", "search"),
            ("Diagnostics", "diagnostics"),
        ]:
            pd.DataFrame(serial(result[key])).round(6).to_excel(
                writer, sheet_name=sheet, index=False
            )
        rows = [
            {"section": "selection", "key": "selected", "value": result["selected"]},
            {
                "section": "provenance",
                "key": "dataset_sha256",
                "value": result["dataset_sha256"],
            },
        ]
        for section in ("config", "timings", "versions"):
            rows += [
                {"section": section, "key": k, "value": json.dumps(serial(v))}
                for k, v in result.get(section, {}).items()
            ]
        rows += [
            {"section": "split", "key": r["flight"], "value": json.dumps(r)}
            for r in result["split"]
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="Config", index=False)
