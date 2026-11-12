"""At most ten new candidates on frozen mixed-valve TRAIN/VAL. TEST unopened."""

import json
import time
from datetime import datetime, timezone
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.null_rate import fit_null_rate, refine_window_rate
from actuator.experiment import evaluate, serial
from run_flightlike import score


def main():
    variation = dict(gain_sd=0.03, bias_sd=0.025, drift_amp=0.02)
    flights, split = pulse_flights(
        seed=621, profile="flightLike", valve_variation=variation
    )
    groups = {s: split[s] for s in ("train", "validation")}
    old = ROOT / "python/audit_results/valve_improvement_20261112_162237"
    prior = json.loads((old / "result.json").read_text())
    assert fingerprint(flights) == prior["dataset_sha256"]
    reference = json.loads((old / "models.json").read_text())[prior["selected"]]
    ddv, angle = reference["ddv"], reference["angle"]
    out = (
        ROOT
        / "python/results/audits"
        / ("valve_search_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    models = {"reference": reference}
    search = []
    candidates = [
        ("quiet_1s", dict(quiet_seconds=1), None),
        ("quiet_2s", dict(quiet_seconds=2), None),
        ("quiet_4s", dict(quiet_seconds=4), None),
        ("quiet_1s_q50", dict(quiet_seconds=1, quiet_quantile=0.5), None),
        ("quiet_2s_q50", dict(quiet_seconds=2, quiet_quantile=0.5), None),
        ("quiet_half_q50", dict(quiet_seconds=0.5, quiet_quantile=0.5), None),
        ("window_2s", {}, 2),
        ("window_10s", {}, 10),
        ("window_30s", {}, 30),
        ("quiet_2s_window", dict(quiet_seconds=2), 10),
    ]
    print("OUT", out, flush=True)
    for i, (name, cfg, window) in enumerate(candidates, 1):
        start = time.perf_counter()
        m = fit_null_rate(
            groups["train"],
            ddv,
            angle,
            adaptive=True,
            tau=2,
            knots=81,
            ridge=0.001,
            observer="stationary",
            gate_source="predicted",
            **cfg,
        )
        if window:
            m = refine_window_rate(m, groups["train"], seconds=window, ridge=0.01)
        fit, rmse = score(m, groups["validation"])
        models[name] = m
        search.append(
            dict(
                iteration=i,
                model=name,
                validation_fit=fit,
                validation_rmse=rmse,
                seconds=time.perf_counter() - start,
                settings=cfg,
                window=window,
            )
        )
        print(i, name, round(fit, 4), flush=True)
        (out / "search.json").write_text(json.dumps(serial(search), indent=2))
        (out / "models.json").write_text(json.dumps(serial(models)))
        if fit >= 90:
            break
    best = min(
        [
            dict(
                model="reference",
                validation_rmse=score(reference, groups["validation"])[1],
            )
        ]
        + search,
        key=lambda r: r["validation_rmse"],
    )["model"]
    per, summary = evaluate(models, groups, include_test=False)
    result = dict(
        selected=best,
        summary=summary,
        per_flight=per,
        search=search,
        variation=variation,
        seed=621,
        dataset_sha256=fingerprint(flights),
        test_evaluated=False,
        split={s: [f.name for f in fs] for s, fs in split.items()},
        stop_reason="target"
        if any(r["validation_fit"] >= 90 for r in search)
        else "10 candidate limit",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
    print("SELECTED", best, out, flush=True)


if __name__ == "__main__":
    main()
