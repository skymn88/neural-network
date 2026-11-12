"""Freeze cross-run validation winners, then evaluate test once without retraining.
Run this only after completing the validation experiment budget.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from actuator.data import ROOT, bundled, pulse_flights, fingerprint
from actuator.experiment import evaluate, diagnostics, serial, write_workbook
from actuator.models import val_rmse


def finalize(ids, destination=None):
    records = []
    for id in ids:
        if not id.replace("_", "").isdigit():
            raise ValueError("Invalid run ID")
        p = ROOT / "python/results" / id
        if not p.is_dir():
            p = ROOT / "python/experiments" / id
        records.append(
            (
                json.loads((p / "result.json").read_text()),
                json.loads((p / "models.json").read_text()),
            )
        )
    source = records[0][0]["config"]["source"]
    if source not in ("bundled", "pulse"):
        raise ValueError("This reference finalizer accepts public synthetic runs only")
    flights, groups = bundled() if source == "bundled" else pulse_flights()
    digest = fingerprint(flights)
    if any(
        r["dataset_sha256"] != digest or r["split"] != records[0][0]["split"]
        for r, _ in records
    ):
        raise ValueError("Runs must use identical data and split")
    models = {}
    provenance = {}
    scores = {}
    for r, ms in records:
        for name, m in ms.items():
            score = r["validation_scores"][name]
            if score is not None and (name not in scores or score < scores[name]):
                scores[name] = score
                models[name] = m
                provenance[name] = {
                    "run": r["id"],
                    "config": r["config"],
                    "timings": r["timings"],
                }
    selected = min(scores, key=scores.get)  # frozen before ANY test inference
    for name, m in models.items():
        actual = val_rmse(m, groups["validation"])
        if abs(actual - scores[name]) > 1e-8 * max(1, abs(actual)):
            raise ValueError("Saved validation score could not be reproduced: " + name)
    start = time.perf_counter()
    per, summary = evaluate(models, groups, include_test=True)
    diag = diagnostics(models, groups["train"])
    elapsed = time.perf_counter() - start
    chosen = next(r for r, _ in records if r["id"] == provenance[selected]["run"])
    result = dict(chosen)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result.update(
        id=stamp,
        selected=selected,
        summary=summary,
        per_flight=per,
        diagnostics=diag,
        validation_scores=scores,
        model_provenance=provenance,
        selection_policy="Best validation checkpoint per family across listed runs, then best validation RAM RMSE; selection frozen before test",
        final_test=True,
        parent_runs=ids,
    )
    result["config"] = dict(chosen["config"], include_test=True)
    result["timings"] = dict(chosen["timings"], final_evaluation=elapsed)
    result["timing_note"] = (
        "total is the selected parent training run; final_evaluation is the additional frozen-model evaluation. Some exploratory runs overlapped in CPU time."
    )
    result["search"] = [
        dict(row, run=r["id"]) for r, _ in records for row in r["search"]
    ]
    out = (destination or ROOT / "python/reference_results") / stamp
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    (out / "models.json").write_text(json.dumps(serial(models)))
    (out / "flights.json").write_text(json.dumps([f.name for f in flights]))
    write_workbook(out, result)
    print(out)
    for r in summary:
        if r["model"] == selected:
            print(r)
    return result, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    finalize(args.runs)
