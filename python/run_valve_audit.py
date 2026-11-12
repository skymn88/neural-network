"""Fixed synthetic stress A/B. No TEST scores or model-default changes."""

import json
import time
from datetime import datetime, timezone
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.models import fit_ddv, fit_angle
from actuator.null_rate import fit_null_rate
from actuator.experiment import evaluate, serial


def main():
    out = (
        ROOT
        / "python/results/audits"
        / ("valve_audit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    rows, per, searches, manifests = [], [], [], []
    scenarios = [
        ("baseline", None),
        ("mixed", dict(gain_sd=0.03, bias_sd=0.025, drift_amp=0.02)),
        ("bias_only", dict(bias_sd=0.04)),
    ]
    for scenario, variation in scenarios:
        flights, groups = pulse_flights(
            seed=621, profile="flightLike", valve_variation=variation
        )
        # Discard TEST before fitting/evaluation. Full split is recorded, never scored.
        manifest = {s: [f.name for f in fs] for s, fs in groups.items()}
        groups = {s: groups[s] for s in ("train", "validation")}
        if variation is None:
            prior = json.loads(
                (
                    ROOT / "python/reference_results/20261112_062616_313440/result.json"
                ).read_text()
            )
            assert fingerprint(flights) == prior["dataset_sha256"]
        manifests.append(
            dict(
                scenario=scenario,
                variation=variation,
                split=manifest,
                dataset_sha256=fingerprint(flights),
            )
        )
        start = time.perf_counter()
        ddv, _ = fit_ddv(groups["train"], groups["validation"])
        angle = fit_angle(groups["train"], groups["validation"])
        for name, observer in [
            ("Rate_slow_null", "legacy"),
            ("Rate_quiet", "stationary"),
            ("Rate_envelope", "envelope"),
        ]:
            t = time.perf_counter()
            model = fit_null_rate(
                groups["train"],
                ddv,
                angle,
                adaptive=True,
                tau=2,
                knots=161,
                observer=observer,
            )
            pp, ss = evaluate({name: model}, groups, include_test=False)
            elapsed = time.perf_counter() - t
            rows.extend(dict(scenario=scenario, **r) for r in ss)
            per.extend(dict(scenario=scenario, **r) for r in pp)
            r = next(r for r in ss if r["stage"] == "RAM_Cascade")
            searches.append(
                dict(
                    scenario=scenario,
                    model=name,
                    seconds=elapsed,
                    validation_fit=r["fit_validation"],
                    validation_rmse=r["rmse_validation"],
                )
            )
            print(scenario, name, round(r["fit_validation"], 3), flush=True)
        manifests[-1]["seconds"] = time.perf_counter() - start
    result = dict(
        summary=rows,
        per_flight=per,
        timings=searches,
        scenarios=manifests,
        test_evaluated=False,
        note="Synthetic stress audit; fixed candidates, no production selection or real-flight claim. Adapted from Claude 177c677.",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(writer, sheet_name="PerFlight", index=False)
        pd.DataFrame(searches).to_excel(writer, sheet_name="Search", index=False)
        pd.DataFrame(
            [dict(scenario=s, settings=json.dumps(v)) for s, v in scenarios]
        ).to_excel(writer, sheet_name="Config", index=False)
    print(out, flush=True)


if __name__ == "__main__":
    main()
