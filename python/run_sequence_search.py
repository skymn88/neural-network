"""20 fixed sequence candidates, validation-only selection, target RAM Fit 90."""

import json
import time
from datetime import datetime, timezone
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.experiment import evaluate, serial
from run_flightlike import score


def main():
    old = ROOT / "python/audit_results/valve_search_20261112_162852"
    prior = json.loads((old / "result.json").read_text())
    reference = json.loads((old / "models.json").read_text())[prior["selected"]]
    flights, split = pulse_flights(
        seed=621, profile="flightLike", valve_variation=prior["variation"]
    )
    assert fingerprint(flights) == prior["dataset_sha256"]
    groups = {s: split[s] for s in ("train", "validation")}
    prepared = prepare_sequences(groups["train"], reference)
    out = (
        ROOT
        / "python/results/audits"
        / ("sequence_search_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    settings = [
        ("tdnn", dict(hidden=24)),
        ("esn", dict(hidden=24, leak=0.2)),
        ("gru", dict(hidden=12)),
        ("lstm", dict(hidden=12)),
        ("tcn", dict(hidden=16)),
        ("elman", dict(hidden=16)),
        ("rbf", dict(hidden=48)),
        ("tdnn", dict(hidden=48, residual=False)),
        ("esn", dict(hidden=48, leak=0.05)),
        ("gru", dict(hidden=24, horizon=512)),
        ("lstm", dict(hidden=24, horizon=512)),
        ("tcn", dict(hidden=24, residual=False)),
        ("elman", dict(hidden=24, residual=False)),
        ("rbf", dict(hidden=96, ridge=0.01)),
        ("tdnn", dict(hidden=48, gate=False)),
        ("esn", dict(hidden=48, leak=0.5, residual=False)),
        ("gru", dict(hidden=24, gate=False)),
        ("lstm", dict(hidden=24, gate=False)),
        ("tcn", dict(hidden=24, gate=False)),
        ("esn", dict(hidden=64, leak=0.1, gate=False)),
    ]
    best_model = reference
    best_name = "reference"
    ref_fit, best_rmse = score(reference, groups["validation"])
    search = []
    print("OUT", out, "REFERENCE", ref_fit, flush=True)
    for i, (family, cfg) in enumerate(settings, 1):
        start = time.perf_counter()
        name = f"{i:02}_{family}"
        try:
            model = fit_sequence(
                prepared, reference, family=family, seed=42, steps=400, **cfg
            )
            fit, rmse = score(model, groups["validation"])
            row = dict(
                iteration=i,
                model=name,
                family=family,
                settings=cfg,
                validation_fit=fit,
                validation_rmse=rmse,
                seconds=time.perf_counter() - start,
                status="ok",
            )
            if rmse < best_rmse:
                best_name, best_model, best_rmse = name, model, rmse
            (out / (name + ".json")).write_text(json.dumps(serial(model)))
        except (ValueError, RuntimeError) as exc:
            row = dict(
                iteration=i,
                model=name,
                family=family,
                settings=cfg,
                seconds=time.perf_counter() - start,
                status="failed",
                error=str(exc),
            )
            fit = float("-inf")
        search.append(row)
        (out / "search.json").write_text(json.dumps(serial(search), indent=2))
        (out / "selected_model.json").write_text(json.dumps(serial(best_model)))
        print(i, name, row.get("validation_fit"), round(row["seconds"], 1), flush=True)
        if fit >= 90:
            break
    per, summary = evaluate(
        {
            best_name: best_model,
            **({"reference": reference} if best_name != "reference" else {}),
        },
        groups,
        include_test=False,
    )
    result = dict(
        selected=best_name,
        summary=summary,
        per_flight=per,
        search=search,
        reference_validation_fit=ref_fit,
        seed=621,
        variation=prior["variation"],
        dataset_sha256=fingerprint(flights),
        split={s: [f.name for f in fs] for s, fs in split.items()},
        test_evaluated=False,
        stop_reason="target"
        if any(r.get("validation_fit", -1e9) >= 90 for r in search)
        else "20 candidate limit",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
    print("SELECTED", best_name, out, flush=True)


if __name__ == "__main__":
    main()
