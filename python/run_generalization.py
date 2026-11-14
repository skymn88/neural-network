"""Frozen 8-flight validation, 3-seed rate ensembles, q999 A/B, 7/14/21 scaling."""

import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from actuator.data import ROOT, fingerprint
from actuator.generalization import (
    PROTOCOL,
    generalization_data,
    cohort_manifest,
    shared_reference,
    training_speed_limit,
    clipped_reference,
    make_rate_ensemble,
)
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.sequence_evaluate import recurrent_states
from actuator.sequence_inference import sequence_features, combine_rate, integrate_rate
from actuator.inference import predict_ddv, angle_values, rollout
from actuator.experiment import metrics, serial


def validation_rank(rows):
    values = [
        r["fit"]
        for r in rows
        if r["split"] == "validation" and r["stage"] == "RAM_Cascade"
    ]
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return float("-inf"), float("-inf")
    return float(min(values)), float(np.mean(values))


def pooled(rows):
    if not rows:
        return dict(fit=None, rmse=None, mean_fit=None, min_fit=None)
    sse = sum(r["sse"] for r in rows)
    sst = sum(r["sst"] for r in rows)
    n = sum(r["samples"] for r in rows)
    fits = [r["fit"] for r in rows]
    return dict(
        fit=float(100 * (1 - np.sqrt(sse / sst))),
        rmse=float(np.sqrt(sse / n)),
        mean_fit=float(np.mean(fits)),
        min_fit=float(min(fits)),
    )


def evaluate_group(models, seeds, train, validation, name):
    first = models[0]
    per = []
    seed_rows = []
    seed_data = []
    for split, flights in [("train", train), ("validation", validation)]:
        for f in flights:
            z = predict_ddv(first["ddv"], f.values[:, 0])
            for stage, input_z in [
                ("RAM_Cascade", z),
                ("RAM_Component", f.values[:, 1]),
            ]:
                features, base = sequence_features(input_z, first["ram"]["reference"])
                rates = []
                for model, seed in zip(models, seeds):
                    spec = model["ram"]["sequence"]
                    raw = (
                        recurrent_states(spec, features) @ np.asarray(spec["head"])
                        + spec["bias"]
                    )
                    rate = combine_rate(spec, raw, base)
                    rates.append(rate)
                    x = integrate_rate(rate, f.values[0, 2], first["ram"]["bounds"])
                    seed_rows.append(
                        dict(
                            model=name,
                            kind="seed_flight",
                            seed=seed,
                            flight=f.name,
                            split=split,
                            stage=stage,
                            **metrics(f.values[32:, 2], x[32:]),
                        )
                    )
                x = integrate_rate(
                    np.mean(rates, axis=0), f.values[0, 2], first["ram"]["bounds"]
                )
                per.append(
                    dict(
                        model=name,
                        flight=f.name,
                        split=split,
                        stage=stage,
                        **metrics(f.values[32:, 2], x[32:]),
                    )
                )
                if stage == "RAM_Cascade":
                    a = angle_values(first["angle"], x)
                    per.append(
                        dict(
                            model=name,
                            flight=f.name,
                            split=split,
                            stage="Angle_Cascade",
                            **metrics(f.values[32:, 3], a[32:]),
                        )
                    )
            for stage, y, p in [
                ("DDV", f.values[:, 1], z),
                (
                    "Angle_Component",
                    f.values[:, 3],
                    angle_values(first["angle"], f.values[:, 2]),
                ),
            ]:
                per.append(
                    dict(
                        model=name,
                        flight=f.name,
                        split=split,
                        stage=stage,
                        **metrics(y[32:], p[32:]),
                    )
                )
    for stage in ("RAM_Cascade", "RAM_Component"):
        macro = []
        worst = []
        for seed in seeds:
            rows = [
                r
                for r in seed_rows
                if r["seed"] == seed
                and r["split"] == "validation"
                and r["stage"] == stage
            ]
            stats = pooled(rows)
            macro.append(stats["mean_fit"])
            worst.append(stats["min_fit"])
            seed_data.append(
                dict(
                    model=name,
                    kind="seed_validation",
                    seed=seed,
                    split="validation",
                    stage=stage,
                    flight="ALL_8",
                    **stats,
                )
            )
        seed_data.append(
            dict(
                model=name,
                kind="seed_variance",
                split="validation",
                stage=stage,
                flight="ALL_8",
                val_min=float(min(macro)),
                val_mean=float(np.mean(macro)),
                val_max=float(max(macro)),
                val_variance=float(np.var(macro, ddof=1)),
                val_std=float(np.std(macro, ddof=1)),
                worst_flight_seed_min=float(min(worst)),
            )
        )
        for f in validation:
            values = [
                r["fit"]
                for r in seed_rows
                if r["split"] == "validation"
                and r["stage"] == stage
                and r["flight"] == f.name
            ]
            seed_data.append(
                dict(
                    model=name,
                    kind="seed_variance",
                    split="validation",
                    stage=stage,
                    flight=f.name,
                    val_min=float(min(values)),
                    val_mean=float(np.mean(values)),
                    val_max=float(max(values)),
                    val_variance=float(np.var(values, ddof=1)),
                )
            )
    return per, seed_rows + seed_data


def main():
    start = time.perf_counter()
    old, split, extra, newval = generalization_data()
    prior_dir = ROOT / "python/audit_results/train_search_20261112_170014"
    prior = json.loads((prior_dir / "result.json").read_text())
    previous = json.loads((prior_dir / "selected_model.json").read_text())
    assert fingerprint(old) == prior["dataset_sha256"]
    validation = split["validation"] + newval
    out = (
        ROOT
        / "python/results/audits"
        / ("generalization_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    manifest = cohort_manifest(old, split, extra, newval)
    manifest["hyperparameters"] = dict(
        family="lstm",
        hidden=24,
        horizon=256,
        steps=800,
        gate_expansion=1.5,
        seed_count=3,
    )
    (out / "protocol.json").write_text(json.dumps(serial(manifest), indent=2))
    print("OUT", out, "TRAIN 7/14/21; VAL 2+6 fixed; TEST closed; 18 fits", flush=True)
    models = {}
    search = []
    per = []
    candidate = []
    summary = []
    training = []
    original_names = {f.name for f in split["train"]}
    for count in PROTOCOL["train_counts"]:
        train = split["train"] + extra[: count - 7]
        ref_start = time.perf_counter()
        ref = shared_reference(train, previous)
        limit = training_speed_limit(train)
        print(
            "REFERENCE",
            count,
            "seconds",
            round(time.perf_counter() - ref_start, 1),
            "q999",
            limit,
            flush=True,
        )
        for clipped in (False, True):
            name = f"n{count:02}_q" + ("999" if clipped else "off")
            common = clipped_reference(ref, limit) if clipped else ref
            prepared = prepare_sequences(train, common)
            members = []
            t = time.perf_counter()
            for seed in PROTOCOL["training_seeds"]:
                fit_start = time.perf_counter()
                m = fit_sequence(
                    prepared,
                    common,
                    family="lstm",
                    hidden=24,
                    horizon=256,
                    steps=800,
                    seed=seed,
                    clip_limit=limit if clipped else None,
                )
                members.append(m)
                training.append(
                    dict(
                        model=name,
                        seed=seed,
                        train_count=count,
                        q999=clipped,
                        seconds=time.perf_counter() - fit_start,
                    )
                )
                (out / (name + "_members.json")).write_text(json.dumps(serial(members)))
                print("FIT", name, seed, round(training[-1]["seconds"], 1), flush=True)
            ensemble = make_rate_ensemble(members, PROTOCOL["training_seeds"])
            models[name] = ensemble
            rows, seed_stats = evaluate_group(
                members, PROTOCOL["training_seeds"], train, validation, name
            )
            per.extend(rows)
            candidate.extend(seed_stats)
            val = [
                r
                for r in rows
                if r["split"] == "validation" and r["stage"] == "RAM_Cascade"
            ]
            tr = [
                r for r in rows if r["split"] == "train" and r["stage"] == "RAM_Cascade"
            ]
            vs, ts = pooled(val), pooled(tr)
            oldtr = pooled([r for r in tr if r["flight"] in original_names])
            nv = pooled([r for r in val if r["flight"].startswith("VAL_NEW_")])
            ov = pooled([r for r in val if not r["flight"].startswith("VAL_NEW_")])
            component = pooled(
                [
                    r
                    for r in rows
                    if r["split"] == "validation" and r["stage"] == "RAM_Component"
                ]
            )
            search.append(
                dict(
                    model=name,
                    train_count=count,
                    seed_count=3,
                    q999=clipped,
                    train_q999_limit=limit,
                    train_min_fit=ts["min_fit"],
                    train_mean_fit=ts["mean_fit"],
                    train_pooled_fit=ts["fit"],
                    original_train_mean_fit=oldtr["mean_fit"],
                    val_min_fit=vs["min_fit"],
                    val_mean_fit=vs["mean_fit"],
                    val_pooled_fit=vs["fit"],
                    val_original_mean_fit=ov["mean_fit"],
                    val_new_mean_fit=nv["mean_fit"],
                    component_val_mean_fit=component["mean_fit"],
                    component_val_min_fit=component["min_fit"],
                    train_val_gap=ts["mean_fit"] - vs["mean_fit"],
                    goal_met=ts["min_fit"] > 95 and vs["min_fit"] > 90,
                    seconds=time.perf_counter() - t,
                )
            )
            for stage in (
                "DDV",
                "RAM_Component",
                "RAM_Cascade",
                "Angle_Component",
                "Angle_Cascade",
            ):
                row = dict(model=name, stage=stage, train_count=count, q999=clipped)
                for s in ("train", "validation"):
                    st = pooled(
                        [r for r in rows if r["split"] == s and r["stage"] == stage]
                    )
                    row.update({k + "_" + s: v for k, v in st.items()})
                summary.append(row)
            print(
                "ENSEMBLE",
                name,
                "VAL min/mean",
                vs["min_fit"],
                vs["mean_fit"],
                "COMP mean",
                component["mean_fit"],
                flush=True,
            )
            for file, value in [
                ("search.json", search),
                ("per_flight.json", per),
                ("candidate_flights.json", candidate),
                ("models.json", models),
            ]:
                (out / file).write_text(json.dumps(serial(value), indent=2))
    selected = max(
        models, key=lambda name: validation_rank([r for r in per if r["model"] == name])
    )
    # Independent portable NumPy check on first validation flight for both pathways.
    from actuator.sequence_evaluate import fast_rollout

    for override in (None, validation[0].values[:, 1]):
        fast = fast_rollout(
            models[selected],
            validation[0].values[:, 0],
            validation[0].values[0, 2],
            override,
        )
        portable = rollout(
            models[selected],
            validation[0].values[:, 0],
            validation[0].values[0, 2],
            override,
        )
        np.testing.assert_allclose(fast, portable, atol=1e-7, rtol=1e-7)
    result = dict(
        selected=selected,
        selection_policy="Ensembles only: max minimum 8-flight validation RAM Cascade Fit, then macro mean; TRAIN not used",
        summary=summary,
        per_flight=per,
        search=search,
        candidate_flights=candidate,
        training=training,
        protocol=manifest,
        test_evaluated=False,
        goal_met=any(r["goal_met"] for r in search),
        timings=dict(total=time.perf_counter() - start),
        notice="All data synthetic. Six new validation flights were fixed before fitting. Validation used for selection, not an independent final test. No real-flight claim.",
    )
    (out / "result.json").write_text(json.dumps(serial(result), indent=2))
    with pd.ExcelWriter(out / "comparison.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(per).to_excel(w, sheet_name="PerFlight", index=False)
        pd.DataFrame(search).to_excel(w, sheet_name="Search", index=False)
        pd.DataFrame(candidate).to_excel(w, sheet_name="CandidateFlights", index=False)
    print("SELECTED", selected, out, flush=True)


if __name__ == "__main__":
    main()
