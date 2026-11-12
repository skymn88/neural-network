"""TRAIN-fit diagnostic search. Validation reported separately; TEST untouched."""

import json
import time
from datetime import datetime, timezone
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.null_rate import fit_null_rate, refine_window_rate
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.sequence_evaluate import fast_score
from actuator.experiment import evaluate, serial


def main():
    old = ROOT / "python/audit_results/sequence_search_20261112_164547"
    prior = json.loads((old / "result.json").read_text())
    previous = json.loads((old / "selected_model.json").read_text())
    flights, split = pulse_flights(
        seed=621, profile="flightLike", valve_variation=prior["variation"]
    )
    assert fingerprint(flights) == prior["dataset_sha256"]
    groups = {s: split[s] for s in ("train", "validation")}
    out = (
        ROOT
        / "python/results/audits"
        / ("train_search_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    settings = [
        ("gru", 1.5, dict(hidden=12, steps=400)),
        ("gru", 2.0, dict(hidden=12, steps=400)),
        ("gru", 3.0, dict(hidden=12, steps=400)),
        ("gru", 1.5, dict(hidden=24, steps=800)),
        ("gru", 2.0, dict(hidden=24, steps=800)),
        ("lstm", 1.5, dict(hidden=24, steps=800)),
        ("lstm", 2.0, dict(hidden=24, steps=800)),
        ("tdnn", 1.5, dict(hidden=48, steps=1000)),
        ("tdnn", 2.0, dict(hidden=48, steps=1000)),
        ("tcn", 1.5, dict(hidden=24, steps=800)),
        ("tcn", 2.0, dict(hidden=24, steps=800)),
        ("gru", 1.5, dict(hidden=24, steps=800, horizon=512)),
        ("gru", 2.0, dict(hidden=24, steps=800, horizon=512)),
        ("gru", 1.5, dict(hidden=48, steps=1000, horizon=512)),
        ("gru", 2.0, dict(hidden=48, steps=1000, horizon=512)),
        ("gru", 1.5, dict(hidden=24, steps=800, gate=False)),
        ("gru", 2.0, dict(hidden=24, steps=800, gate=False)),
        ("lstm", 2.0, dict(hidden=48, steps=1000, horizon=512)),
        ("tdnn", 2.0, dict(hidden=64, steps=1500, residual=False)),
        ("gru", 2.0, dict(hidden=48, steps=1500, residual=False)),
    ]
    best = previous
    best_name = "previous_gru"
    best_fit, best_rmse = fast_score(previous, groups["train"])
    search = []
    cache = {}
    print("OUT", out, "REFERENCE TRAIN", best_fit, flush=True)
    for i, (family, expansion, cfg) in enumerate(settings, 1):
        start = time.perf_counter()
        name = f"{i:02}_{family}_gate{expansion:g}"
        if expansion not in cache:
            ref = fit_null_rate(
                groups["train"],
                previous["ddv"],
                previous["angle"],
                adaptive=True,
                tau=2,
                knots=81,
                ridge=0.001,
                observer="stationary",
                gate_source="predicted",
                gate_expansion=expansion,
            )
            ref = refine_window_rate(ref, groups["train"], seconds=2, ridge=0.01)
            cache[expansion] = (ref, prepare_sequences(groups["train"], ref))
        ref, prepared = cache[expansion]
        model = fit_sequence(prepared, ref, family=family, seed=42, **cfg)
        train_fit, train_rmse = fast_score(model, groups["train"])
        val_fit, val_rmse = fast_score(model, groups["validation"])
        row = dict(
            iteration=i,
            model=name,
            family=family,
            gate_expansion=expansion,
            settings=cfg,
            train_fit=train_fit,
            train_rmse=train_rmse,
            validation_fit=val_fit,
            validation_rmse=val_rmse,
            seconds=time.perf_counter() - start,
        )
        search.append(row)
        if train_rmse < best_rmse:
            best, best_name, best_rmse = model, name, train_rmse
        (out / (name + ".json")).write_text(json.dumps(serial(model)))
        (out / "search.json").write_text(json.dumps(serial(search), indent=2))
        (out / "selected_model.json").write_text(json.dumps(serial(best)))
        print(
            i,
            name,
            "TRAIN",
            round(train_fit, 4),
            "VAL",
            round(val_fit, 4),
            "sec",
            round(row["seconds"], 1),
            flush=True,
        )
        if train_fit > 90:
            break
    # Final metrics computed again with portable NumPy rollout, including Component.
    per, summary = evaluate(
        {best_name: best, "previous_gru": previous}, groups, include_test=False
    )
    portable = next(
        r for r in summary if r["model"] == best_name and r["stage"] == "RAM_Cascade"
    )
    accelerated = fast_score(best, groups["train"])[0]
    assert abs(portable["fit_train"] - accelerated) < 1e-5
    result = dict(
        selected=best_name,
        selection_basis="TRAIN RAM Cascade RMSE; diagnostic, not validation-selected production replacement",
        summary=summary,
        per_flight=per,
        search=search,
        seed=621,
        variation=prior["variation"],
        dataset_sha256=fingerprint(flights),
        split={s: [f.name for f in fs] for s, fs in split.items()},
        test_evaluated=False,
        stop_reason="TRAIN >90"
        if any(r["train_fit"] > 90 for r in search)
        else "20 candidate limit",
        portable_score_checked=True,
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
    print("SELECTED", best_name, out, flush=True)


if __name__ == "__main__":
    main()
