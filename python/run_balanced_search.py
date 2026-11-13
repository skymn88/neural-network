"""At most 20 candidates: EVERY train flight Fit>90 and pooled validation Fit>75."""

import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from actuator.data import ROOT, pulse_flights, fingerprint
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.sequence_evaluate import fast_rollout
from actuator.head_refine import refine_sequence_head
from actuator.experiment import metrics, evaluate, serial


def flight_report(model, flights):
    rows = []
    for f in flights:
        x = fast_rollout(model, f.values[:, 0], f.values[0, 2])[1]
        rows.append(dict(flight=f.name, **metrics(f.values[32:, 2], x[32:])))
    if any(not np.isfinite(r["rmse"]) for r in rows):
        return rows, float("-inf"), float("-inf")
    pooled = 100 * (
        1 - np.sqrt(sum(r["sse"] for r in rows) / sum(r["sst"] for r in rows))
    )
    worst = min(r["fit"] if r["fit"] is not None else float("-inf") for r in rows)
    return rows, float(pooled), float(worst)


def meets_goal(train_worst, validation_fit):
    return bool(train_worst > 90 and validation_fit > 75)


def main():
    prior_dir = ROOT / "python/audit_results/train_search_20261112_170014"
    prior = json.loads((prior_dir / "result.json").read_text())
    previous = json.loads((prior_dir / "selected_model.json").read_text())
    flights, split = pulse_flights(
        seed=621, profile="flightLike", valve_variation=prior["variation"]
    )
    assert fingerprint(flights) == prior["dataset_sha256"]
    groups = {s: split[s] for s in ("train", "validation")}
    reference = dict(
        ddv=previous["ddv"], angle=previous["angle"], ram=previous["ram"]["reference"]
    )
    prepared = prepare_sequences(groups["train"], reference)
    old_rows, old_pooled, old_worst = flight_report(previous, groups["train"])
    _, old_val, _ = flight_report(previous, groups["validation"])
    # Flight weights derive only from baseline TRAIN errors, never validation.
    weights = np.maximum([1 - r["fit"] / 100 for r in old_rows], 0.02) ** 2
    weights = np.clip(weights / weights.mean(), 0.5, 3).tolist()
    settings = [
        ("tcn", dict(hidden=24, steps=800)),
        ("elman", dict(hidden=24, steps=800)),
        ("rbf", dict(hidden=96, ridge=0.001)),
        ("head", dict(iterations=3, ridge=0.01, focus=1)),
        ("lstm", dict(hidden=24, steps=1200)),
        ("lstm", dict(hidden=24, steps=1200, flight_weights=weights)),
        ("lstm", dict(hidden=32, steps=1200, flight_weights=weights)),
        ("head", dict(iterations=8, ridge=0.001, focus=2)),
        ("lstm", dict(hidden=24, steps=1000, horizon=512, flight_weights=weights)),
        (
            "lstm",
            dict(
                hidden=32,
                steps=1600,
                horizon=512,
                flight_weights=weights,
                position_weight=1.0,
            ),
        ),
        ("tcn", dict(hidden=48, steps=1600, flight_weights=weights)),
        ("tdnn", dict(hidden=64, steps=1600, flight_weights=weights)),
        ("rbf", dict(hidden=192, ridge=0.01)),
        ("elman", dict(hidden=48, steps=1600, flight_weights=weights)),
        ("lstm", dict(hidden=48, steps=1600, flight_weights=weights)),
        ("head", dict(iterations=12, ridge=0.01, focus=3)),
        ("gru", dict(hidden=32, steps=1600, horizon=512, flight_weights=weights)),
        ("lstm", dict(hidden=32, steps=1600, seed=43, flight_weights=weights)),
        (
            "lstm",
            dict(
                hidden=48,
                steps=2000,
                horizon=512,
                flight_weights=weights,
                position_weight=2.0,
            ),
        ),
        ("head_best", dict(iterations=10, ridge=0.005, focus=2)),
    ]
    out = (
        ROOT
        / "python/results/audits"
        / ("balanced_search_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(serial(settings), indent=2))
    best = previous
    best_name = "previous_lstm"
    best_worst = old_worst
    best_any = previous
    any_worst = old_worst
    search = []
    all_per = []
    print("OUT", out, "REFERENCE", old_pooled, old_worst, old_val, flush=True)
    for i, (family, cfg) in enumerate(settings, 1):
        start = time.perf_counter()
        name = f"{i:02}_{family}"
        (out / "progress.json").write_text(
            json.dumps(dict(iteration=i, model=name, status="running"))
        )
        try:
            if family.startswith("head"):
                model = refine_sequence_head(
                    best_any if family == "head_best" else previous,
                    groups["train"],
                    **cfg,
                )
            else:
                kw = dict(seed=42)
                kw.update(cfg)
                model = fit_sequence(prepared, reference, family=family, **kw)
            tr, pool, worst = flight_report(model, groups["train"])
            va, vfit, vworst = flight_report(model, groups["validation"])
            row = dict(
                iteration=i,
                model=name,
                family=family,
                settings=cfg,
                train_fit=pool,
                train_min_fit=worst,
                validation_fit=vfit,
                validation_min_fit=vworst,
                seconds=time.perf_counter() - start,
                status="ok",
                goal_met=meets_goal(worst, vfit),
            )
            all_per.extend(
                dict(model=name, split=s, **r)
                for s, rows in [("train", tr), ("validation", va)]
                for r in rows
            )
            if worst > any_worst:
                best_any, any_worst = model, worst
            if vfit > 75 and worst > best_worst:
                best, best_name, best_worst = model, name, worst
            (out / (name + ".json")).write_text(json.dumps(serial(model)))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row = dict(
                iteration=i,
                model=name,
                family=family,
                settings=cfg,
                status="failed",
                error=str(exc),
                seconds=time.perf_counter() - start,
                goal_met=False,
            )
        search.append(row)
        (out / "search.json").write_text(json.dumps(serial(search), indent=2))
        (out / "selected_model.json").write_text(json.dumps(serial(best)))
        print(
            i,
            name,
            "TRAIN",
            row.get("train_fit"),
            "MIN",
            row.get("train_min_fit"),
            "VAL",
            row.get("validation_fit"),
            "secs",
            round(row["seconds"], 1),
            flush=True,
        )
        if row["goal_met"]:
            break
    per, summary = evaluate(
        {
            best_name: best,
            **({"previous_lstm": previous} if best_name != "previous_lstm" else {}),
        },
        groups,
        include_test=False,
    )
    # Portable NumPy audit of selected model versus accelerated selection scores.
    actual = [
        r["fit"]
        for r in per
        if r["model"] == best_name
        and r["stage"] == "RAM_Cascade"
        and r["split"] == "train"
    ]
    assert abs(min(actual) - best_worst) < 1e-5
    result = dict(
        selected=best_name,
        summary=summary,
        per_flight=per,
        search=search,
        candidate_per_flight=all_per,
        selection_policy="maximize minimum TRAIN flight RAM Cascade Fit subject to pooled validation Fit>75; stop every TRAIN>90",
        goal_met=any(r["goal_met"] for r in search),
        test_evaluated=False,
        seed=621,
        variation=prior["variation"],
        dataset_sha256=fingerprint(flights),
        split={s: [f.name for f in fs] for s, fs in split.items()},
        portable_score_checked=True,
        stop_reason="goal"
        if any(r["goal_met"] for r in search)
        else "20 candidate limit",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
        pd.DataFrame(all_per).to_excel(w, sheet_name="CandidateFlights", index=False)
    (out / "progress.json").write_text(
        json.dumps(dict(status="done", selected=best_name))
    )
    print("SELECTED", best_name, out, flush=True)


if __name__ == "__main__":
    main()
