"""Three bounded TRAIN/VAL trials for mixed valve variation; TEST stays closed."""

import json
import time
from datetime import datetime, timezone
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.models import fit_ddv, fit_angle
from actuator.null_rate import fit_null_rate
from actuator.experiment import evaluate, serial


def main():
    variation = dict(gain_sd=0.03, bias_sd=0.025, drift_amp=0.02)
    flights, split = pulse_flights(
        seed=621, profile="flightLike", valve_variation=variation
    )
    groups = {s: split[s] for s in ("train", "validation")}
    ddv, _ = fit_ddv(groups["train"], groups["validation"])
    angle = fit_angle(groups["train"], groups["validation"])
    models, search, summary, per = {}, [], [], []
    # Fixed existing reference plus exactly three new, predeclared candidates.
    settings = [
        ("Rate_quiet_reference", dict(observer="stationary")),
        ("Rate_predicted_gate", dict(observer="stationary", gate_source="predicted")),
        ("Rate_predicted_envelope", dict(observer="envelope", gate_source="predicted")),
        (
            "Rate_predicted_smooth",
            dict(observer="stationary", gate_source="predicted", knots=81, ridge=0.001),
        ),
    ]
    for name, cfg in settings:
        t = time.perf_counter()
        kw = dict(adaptive=True, tau=2, knots=161)
        kw.update(cfg)
        model = fit_null_rate(groups["train"], ddv, angle, **kw)
        pp, ss = evaluate({name: model}, groups, include_test=False)
        r = next(r for r in ss if r["stage"] == "RAM_Cascade")
        search.append(
            dict(
                model=name,
                seconds=time.perf_counter() - t,
                validation_fit=r["fit_validation"],
                validation_rmse=r["rmse_validation"],
                settings=kw,
            )
        )
        models[name] = model
        summary.extend(ss)
        per.extend(pp)
        print(name, round(r["fit_validation"], 4), flush=True)
    selected = min(search, key=lambda r: r["validation_rmse"])["model"]
    out = (
        ROOT
        / "python/results/audits"
        / ("valve_improvement_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    result = dict(
        selected=selected,
        summary=summary,
        per_flight=per,
        search=search,
        variation=variation,
        seed=621,
        dataset_sha256=fingerprint(flights),
        split={s: [f.name for f in fs] for s, fs in split.items()},
        test_evaluated=False,
        note="Three new validation trials; reference retained. Synthetic only; defaults unchanged.",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    (out / "models.json").write_text(json.dumps(serial(models)))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
    print(out, flush=True)


if __name__ == "__main__":
    main()
