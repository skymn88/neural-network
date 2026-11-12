"""Three predeclared RAM trials on frozen flightLike TRAIN/VAL; no test selection."""

import json
import time
import platform
from datetime import datetime, timezone
import numpy as np
from actuator.data import ROOT, Flight, pulse_flights, fingerprint
from actuator.null_rate import fit_null_rate, refine_window_rate
from actuator.experiment import evaluate, serial, write_workbook
from run_flightlike import score


def main():
    old = ROOT / "python/reference_results/20261112_062616_313440"
    prior = json.loads((old / "result.json").read_text())
    baseline = json.loads((old / "models.json").read_text())[prior["selected"]]
    if (old / "flights.npz").exists():
        with np.load(old / "flights.npz", allow_pickle=False) as raw:
            flights = [
                Flight(name, raw[f"f{i}"][:, 0], raw[f"f{i}"][:, 1:])
                for i, name in enumerate(json.loads((old / "flights.json").read_text()))
            ]
    else:
        flights, _ = pulse_flights(seed=621, profile="flightLike")
    assert fingerprint(flights) == prior["dataset_sha256"]
    lookup = {f.name: f for f in flights}
    groups = {
        s: [lookup[r["flight"]] for r in prior["split"] if r["split"] == s]
        for s in ("train", "validation", "test")
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "python/results" / stamp
    out.mkdir(parents=True)
    models = {"Rate_previous": baseline}
    trials = []
    start = time.perf_counter()
    quiet_model = None
    for i, (name, observer, refine) in enumerate(
        [
            ("Rate_quiet", "stationary", False),
            ("Rate_envelope", "envelope", False),
            ("Rate_windowOE", "stationary", True),
        ],
        1,
    ):
        t = time.perf_counter()
        if refine:
            m = refine_window_rate(quiet_model, groups["train"], seconds=5, ridge=0.01)
        else:
            m = fit_null_rate(
                groups["train"],
                baseline["ddv"],
                baseline["angle"],
                adaptive=True,
                knots=161,
                tau=2,
                observer=observer,
            )
        if name == "Rate_quiet":
            quiet_model = m
        fit, rmse = score(m, groups["validation"])
        seconds = time.perf_counter() - t
        trials.append(
            {
                "iteration": i,
                "model": name,
                "validation_fit": fit,
                "validation_rmse": rmse,
                "seconds": seconds,
                "observer": observer,
                "window_refinement": refine,
            }
        )
        models[name] = m
        print(
            f"TRIAL {i}/3 {name}: VAL Fit={fit:.4f}, RMSE={rmse:.5f}, {seconds:.2f}s",
            flush=True,
        )
        (out / "models.json").write_text(json.dumps(serial(models)))
        (out / "search_progress.json").write_text(json.dumps(serial(trials), indent=2))
        if fit >= 95:
            break
    scores = {name: score(m, groups["validation"])[1] for name, m in models.items()}
    selected = min(scores, key=scores.get)
    print(
        "FROZEN selection:",
        selected,
        "test now reported (historical test already seen in prior turn)",
        flush=True,
    )
    per, summary = evaluate(models, groups, include_test=True)
    # Additional unseen synthetic audit generated only after freezing; not used to tune.
    fresh, _ = pulse_flights(seed=9621, durations=(2400, 2700), profile="flightLike")
    audit_per, audit_summary = evaluate(
        {selected: models[selected]}, {"test": fresh}, include_test=True
    )
    result = {
        "id": stamp,
        "selected": selected,
        "config": dict(prior["config"], max_iterations=3, observer_search=True),
        "summary": summary,
        "per_flight": per,
        "search": [dict(t, stage="RAM") for t in trials],
        "diagnostics": [
            dict(r, diagnostic="FreshSyntheticAudit", audit_seed=9621)
            for r in audit_per
        ],
        "fresh_test_summary": audit_summary,
        "fresh_test_sha256": fingerprint(fresh),
        "split": prior["split"],
        "dataset_sha256": prior["dataset_sha256"],
        "validation_scores": scores,
        "timings": {
            **{t["model"]: t["seconds"] for t in trials},
            "total": time.perf_counter() - start,
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "trials": trials,
        "notice": "Synthetic flightLike comparison; historical test is reused. Separate seed-9621 audit generated after selection.",
        "selection_policy": "Validation only; stop at 95 Fit or three candidates. Previous model retained if better.",
    }
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    (out / "flights.json").write_text(json.dumps([f.name for f in flights]))
    write_workbook(out, result)
    print("RESULT", out, flush=True)
    for row in summary:
        if row["model"] == selected:
            print(row, flush=True)
    print("FRESH_AUDIT", audit_summary, flush=True)


if __name__ == "__main__":
    main()
