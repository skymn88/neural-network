"""Three additional settings, three seeds; frozen 8-flight validation, TEST closed."""

import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from actuator.data import ROOT, fingerprint
from actuator.generalization import generalization_data, make_rate_ensemble
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.experiment import serial, write_workbook
from run_generalization import evaluate_group, pooled, validation_rank


def eligible_models(models, per, candidates):
    eligible = []
    for name in models:
        rows = [
            r
            for r in per
            if r["model"] == name
            and r["stage"] == "RAM_Cascade"
            and r["split"] in ("train", "validation")
        ]
        rows += [
            r
            for r in candidates
            if r.get("model") == name
            and r.get("kind") == "seed_flight"
            and r.get("split") == "validation"
            and r.get("stage") == "RAM_Cascade"
        ]
        if rows and all(
            r["fit"] is not None and np.isfinite(r["fit"]) and r["fit"] >= 0
            for r in rows
        ):
            eligible.append(name)
    return eligible


def main():
    start = time.perf_counter()
    prior = ROOT / "python/reference_results/20261113_051726_000000"
    old = json.loads((prior / "result.json").read_text())
    baseline = json.loads((prior / "models.json").read_text())[old["selected"]]
    _, groups, extra, newval = generalization_data()
    train = groups["train"] + extra[:7]
    val = groups["validation"] + newval
    common = dict(
        ddv=baseline["ddv"],
        angle=baseline["angle"],
        ram=baseline["ram"]["members"][0]["reference"],
    )
    prepared = prepare_sequences(train, common)
    grid = [
        dict(family="gru", hidden=24, steps=800, horizon=256, position_weight=0.2),
        dict(family="lstm", hidden=24, steps=1000, horizon=512, position_weight=0.8),
        dict(family="lstm", hidden=32, steps=800, horizon=256, position_weight=0.5),
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "python/results" / stamp
    out.mkdir(parents=True)
    print("OUT", out, flush=True)
    models = {"reference": baseline}
    per = [
        dict(r, model="reference")
        for r in old["per_flight"]
        if r["model"] == old["selected"]
    ]
    candidates = []
    search = []
    summary = []
    seeds = [42, 43, 44]
    for i, cfg in enumerate(grid, 1):
        name = f"{i:02}_{cfg['family']}"
        t = time.perf_counter()
        members = []
        for seed in seeds:
            members.append(
                fit_sequence(
                    prepared,
                    common,
                    seed=seed,
                    clip_limit=common["ram"]["map_rate_limit"],
                    **cfg,
                )
            )
            print("FIT", name, seed, round(time.perf_counter() - t, 1), flush=True)
        models[name] = make_rate_ensemble(members, seeds)
        rows, records = evaluate_group(members, seeds, train, val, name)
        per.extend(rows)
        candidates.extend(records)
        values = [
            r
            for r in rows
            if r["stage"] == "RAM_Cascade" and r["split"] == "validation"
        ]
        stats = pooled(values)
        search.append(dict(model=name, **cfg, **stats, seconds=time.perf_counter() - t))
        for filename, value in [
            ("models.json", models),
            ("per_flight.json", per),
            ("search.json", search),
            ("candidate_flights.json", candidates),
        ]:
            (out / filename).write_text(json.dumps(serial(value), indent=2))
        print("RESULT", name, stats, flush=True)
    selected = max(
        eligible_models(models, per, candidates),
        key=lambda name: validation_rank([r for r in per if r["model"] == name]),
    )
    for name in models:
        for stage in [
            "DDV",
            "RAM_Component",
            "RAM_Cascade",
            "Angle_Component",
            "Angle_Cascade",
        ]:
            row = dict(model=name, stage=stage)
            for split in ["train", "validation"]:
                stats = pooled(
                    [
                        r
                        for r in per
                        if r["model"] == name
                        and r["stage"] == stage
                        and r["split"] == split
                    ]
                )
                row.update({f"{k}_{split}": v for k, v in stats.items()})
            summary.append(row)
    result = dict(
        id=stamp,
        selected=selected,
        config=dict(
            source="mixedGeneralizationValidation",
            seed=621,
            include_test=False,
            selectionBasis="validation",
        ),
        summary=summary,
        per_flight=per,
        search=search,
        candidate_flights=candidates,
        diagnostics=[],
        split=[
            dict(flight=f.name, split="validation", samples=len(f.time)) for f in val
        ],
        dataset_sha256=fingerprint(val),
        timings=dict(total=time.perf_counter() - start),
        validation_scores={
            name: validation_rank([r for r in per if r["model"] == name])[1]
            for name in models
        },
        notice="Synthetic only; frozen 8-flight validation; TEST closed. Selection: worst flight Fit then macro mean. Export candidate, not real-flight qualification.",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    (out / "flights.json").write_text(json.dumps([f.name for f in val]))
    write_workbook(out, result)
    with pd.ExcelWriter(
        out / "comparison.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        pd.DataFrame(serial(candidates)).to_excel(
            writer, sheet_name="CandidateFlights", index=False
        )
    print(
        "SELECTED",
        selected,
        validation_rank([r for r in per if r["model"] == selected]),
        flush=True,
    )


if __name__ == "__main__":
    main()
