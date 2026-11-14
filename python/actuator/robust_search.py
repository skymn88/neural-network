"""Opt-in bounded validation/LOFO search. TEST is never evaluated here."""

import copy
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import ROOT, fingerprint
from .experiment import metrics, serial, write_workbook
from .models import fit_ddv, fit_angle, fit_oe, fit_neural, fit_stable
from .null_rate import fit_null_rate, refine_window_rate
from .sequence_train import prepare_sequences, fit_sequence
from .sequence_evaluate import fast_rollout
from .generalization import training_speed_limit, clipped_reference
from .seed_ensemble import make_ensemble
from .inference import angle_values

GRID = [
    dict(family=f, hidden=h, horizon=z, steps=s)
    for h, z, s in [(12, 128, 200), (24, 256, 400)]
    for f in ("null_rate", "gru_rate", "lstm_rate", "oe", "neural", "stable")
]


class SearchStopped(Exception):
    pass


def selection_folds(groups):
    if len(groups["validation"]) >= 3:
        return [(groups["train"], groups["validation"])], "validation"
    train = groups["train"]
    if len(train) < 2 or len({f.name for f in train}) != len(train):
        raise ValueError(
            "Reliable selection needs at least two distinct TRAIN flights for LOFO"
        )
    return [
        ([f for j, f in enumerate(train) if j != i], [held])
        for i, held in enumerate(train)
    ], "leave-one-flight-out"


def score_model(model, flights, candidate, seed, split, check=lambda: None):
    rows = []
    for f in flights:
        check()
        d, x, a = fast_rollout(model, f.values[:, 0], f.values[0, 2])
        xc = fast_rollout(
            model, f.values[:, 0], f.values[0, 2], ddv_override=f.values[:, 1]
        )[1]
        for stage, truth, pred in [
            ("DDV", f.values[:, 1], d),
            ("RAM_Component", f.values[:, 2], xc),
            ("RAM_Cascade", f.values[:, 2], x),
            (
                "Angle_Component",
                f.values[:, 3],
                angle_values(model["angle"], f.values[:, 2]),
            ),
            ("Angle_Cascade", f.values[:, 3], a),
        ]:
            row = dict(
                model=candidate,
                seed=seed,
                flight=f.name,
                stage=stage,
                split=split,
                **metrics(truth[32:], pred[32:]),
            )
            # Nonfinite warm-up cannot be hidden by the metric's initial guard.
            row["finite_rollout"] = bool(np.isfinite(pred).all())
            rows.append(row)
    return rows


def flight_stats(rows, stage="RAM_Cascade"):
    rs = [r for r in rows if r["stage"] == stage]
    bad = [
        r["flight"]
        for r in rs
        if r["fit"] is None
        or not np.isfinite(r["fit"])
        or r["fit"] < 0
        or not r.get("finite_rollout", True)
    ]
    fits = [r["fit"] for r in rs if r["fit"] is not None and np.isfinite(r["fit"])]
    return dict(
        min_fit=min(fits) if fits else None,
        mean_fit=float(np.mean(fits)) if fits else None,
        max_fit=max(fits) if fits else None,
        rejected_flights=bad,
    )


def rank_key(row):
    return (row["min_fit"], row["mean_fit"]) if row["eligible"] else (-np.inf, -np.inf)


def fit_candidate(train, cfg, seed, check, cache):
    check()
    key = tuple(f.name for f in train)
    if key not in cache:
        ddv, _ = fit_ddv(train, train, fixed_ridge=1e-3)
        angle = fit_angle(train, train)
        ref = fit_null_rate(
            train,
            ddv,
            angle,
            adaptive=True,
            tau=2,
            observer="stationary",
            gate_source="predicted",
            gate_expansion=1.5,
        )
        ref = refine_window_rate(ref, train, seconds=2, ridge=0.01)
        cache[key] = (ddv, angle, ref)
    ddv, angle, ref = cache[key]
    check()
    limit = training_speed_limit(train) if cfg.get("clip") else None
    if limit is not None:
        ref = clipped_reference(ref, limit)
    family = cfg["family"]
    if family == "null_rate":
        return copy.deepcopy(ref)
    if family in ("gru_rate", "lstm_rate"):
        prepared = prepare_sequences(train, ref)
        check()
        return fit_sequence(
            prepared,
            ref,
            family=family.split("_")[0],
            hidden=cfg["hidden"],
            steps=cfg["steps"],
            horizon=cfg["horizon"],
            seed=seed,
            clip_limit=limit,
            check_cancel=check,
        )
    kwargs = dict(train=train, val=train, ddv=ddv, angle=angle, check_cancel=check)
    if family == "oe":
        return fit_oe(**kwargs)[0]
    kwargs.update(
        steps=cfg["steps"],
        horizon=cfg["horizon"],
        hidden=cfg["hidden"],
        seed=seed,
        log=lambda _: check(),
    )
    if family == "neural":
        return fit_neural(**kwargs)[0]["NN_joint"]
    if family == "stable":
        return fit_stable(**kwargs)[0]
    raise ValueError("Unsupported search family")


def summarize(per, models):
    rows = []
    for model in models:
        for stage in (
            "DDV",
            "RAM_Component",
            "RAM_Cascade",
            "Angle_Component",
            "Angle_Cascade",
        ):
            row = dict(model=model, stage=stage)
            for split in ("train", "validation"):
                rs = [
                    r
                    for r in per
                    if r["model"] == model
                    and r["stage"] == stage
                    and r["split"] == split
                ]
                if rs:
                    sse = sum(r["sse"] for r in rs)
                    sst = sum(r["sst"] for r in rs)
                    n = sum(r["samples"] for r in rs)
                    row["fit_" + split] = (
                        100 * (1 - np.sqrt(sse / sst)) if sst > 1e-18 else None
                    )
                    row["rmse_" + split] = np.sqrt(sse / n)
            rows.append(row)
    return rows


def search(
    flights,
    groups,
    source="user",
    max_candidates=6,
    max_minutes=30,
    seed=42,
    methods=None,
    clip=False,
    log=print,
    cancel=lambda: False,
    out=None,
    trainer=fit_candidate,
):
    start = time.perf_counter()
    deadline = start + 60 * max_minutes
    folds, basis = selection_folds(groups)
    grid = [
        dict(c, clip=clip) for c in GRID if methods is None or c["family"] in methods
    ][:max_candidates]
    if not grid:
        raise ValueError("No candidate family selected")
    seeds = [seed, seed + 1, seed + 2]
    out = out or ROOT / "python/results" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    candidate_rows = []
    models = {}
    per = []
    cache = {}
    stopped = None

    def check():
        if cancel():
            raise SearchStopped("Cancelled by user")
        if time.perf_counter() >= deadline:
            raise SearchStopped("Minute limit reached")

    def checkpoint():
        payload = dict(
            search=candidates,
            candidate_flights=candidate_rows,
            stopped=stopped,
            selection_basis=basis,
            test_evaluated=False,
        )
        temp = out / "search.partial.tmp"
        temp.write_text(json.dumps(serial(payload), indent=2), encoding="utf8")
        temp.replace(out / "search.partial.json")
        temp = out / "models.partial.tmp"
        temp.write_text(json.dumps(serial(models)), encoding="utf8")
        temp.replace(out / "models.partial.json")

    log(
        f"Reliable selection: {basis}; {len(grid)} configurations × 3 seeds; TEST closed"
    )
    if basis != "validation":
        log(
            "Notice: validation flights <3; TRAIN selection replaced by leave-one-flight-out over TRAIN flights. Original split unchanged."
        )
    for ci, cfg in enumerate(grid):
        name = f"{ci + 1:02}_{cfg['family']}_h{cfg['hidden']}"
        t = time.perf_counter()
        valrows = []
        seedrows = []
        fullmembers = None
        try:
            for fi, (train, held) in enumerate(folds):
                members = []
                for training_seed in seeds:
                    check()
                    log(
                        f"Candidate {ci + 1}/{len(grid)} {name}; fold {fi + 1}/{len(folds)}; seed {training_seed}"
                    )
                    model = trainer(train, cfg, training_seed, check, cache)
                    members.append(model)
                    rows = score_model(
                        model, held, name, training_seed, "validation", check
                    )
                    for r in rows:
                        r.update(
                            record="seed_flight", fold=fi + 1, validation_source=basis
                        )
                    seedrows.extend(rows)
                    candidate_rows.extend(rows)
                    checkpoint()
                ensemble = make_ensemble(members, seeds)
                rows = score_model(
                    ensemble, held, name, "ensemble", "validation", check
                )
                for r in rows:
                    r.update(
                        record="ensemble_flight", fold=fi + 1, validation_source=basis
                    )
                valrows.extend(rows)
                candidate_rows.extend(rows)
                if basis == "validation":
                    fullmembers = members
            stats = flight_stats(valrows)
            # Reject any negative/nonfinite free-run ensemble flight; retain evidence.
            seed_bad = flight_stats(seedrows)["rejected_flights"]
            stats["rejected_flights"] = sorted(
                set(stats["rejected_flights"] + seed_bad)
            )
            eligible = not stats["rejected_flights"] and stats["min_fit"] is not None
            seed_stats = []
            for training_seed in seeds:
                st = flight_stats([r for r in seedrows if r["seed"] == training_seed])
                seed_stats.append(
                    st["mean_fit"] if st["mean_fit"] is not None else float("nan")
                )
                candidate_rows.append(
                    dict(model=name, record="seed_validation", seed=training_seed, **st)
                )
            candidate_rows.append(
                dict(
                    model=name,
                    record="seed_variance",
                    val_seed_min=float(np.min(seed_stats)),
                    val_seed_mean=float(np.mean(seed_stats)),
                    val_seed_max=float(np.max(seed_stats)),
                    val_seed_std=float(np.std(seed_stats)),
                )
            )
            for flight in sorted({r["flight"] for r in seedrows}):
                for stage in ("RAM_Component", "RAM_Cascade"):
                    values = [
                        r["fit"]
                        for r in seedrows
                        if r["flight"] == flight and r["stage"] == stage
                    ]
                    values = [v for v in values if v is not None and np.isfinite(v)]
                    candidate_rows.append(
                        dict(
                            model=name,
                            record="seed_flight_variance",
                            flight=flight,
                            stage=stage,
                            val_seed_min=min(values) if values else None,
                            val_seed_mean=float(np.mean(values)) if values else None,
                            val_seed_max=max(values) if values else None,
                            val_seed_std=float(np.std(values)) if values else None,
                        )
                    )
            if eligible and fullmembers is None:
                fullmembers = [
                    trainer(groups["train"], cfg, s, check, cache) for s in seeds
                ]
            trainrows = []
            if fullmembers is not None:
                ensemble = make_ensemble(fullmembers, seeds)
                trainrows = score_model(
                    ensemble, groups["train"], name, "ensemble", "train", check
                )
                models[name] = ensemble
            tr = flight_stats(trainrows)
            stats["rejected_flights"] = sorted(
                set(stats["rejected_flights"] + tr["rejected_flights"])
            )
            eligible = eligible and not tr["rejected_flights"]
            comp = flight_stats(valrows, "RAM_Component")
            gap = (
                tr["mean_fit"] - stats["mean_fit"]
                if tr["mean_fit"] is not None and stats["mean_fit"] is not None
                else None
            )
            row = dict(
                model=name,
                **cfg,
                **stats,
                eligible=eligible,
                train_mean_fit=tr["mean_fit"],
                train_val_gap=gap,
                gap_warning=gap is not None and gap > 25,
                component_mean_fit=comp["mean_fit"],
                component_min_fit=comp["min_fit"],
                seconds=time.perf_counter() - t,
            )
            candidates.append(row)
            per.extend(trainrows + valrows)
            log(
                f"{name}: minimum/mean validation Fit {stats['min_fit']}/{stats['mean_fit']}; rejected flights: {stats['rejected_flights']}"
            )
            checkpoint()
        except SearchStopped as e:
            stopped = str(e)
            checkpoint()
            log(stopped)
            break
        except Exception as e:
            candidates.append(
                dict(
                    model=name,
                    **cfg,
                    eligible=False,
                    min_fit=None,
                    mean_fit=None,
                    error=str(e),
                    seconds=time.perf_counter() - t,
                )
            )
            checkpoint()
            log(f"{name}: failed: {e}")
    eligible = [r for r in candidates if r["eligible"] and r["model"] in models]
    selected = max(eligible, key=rank_key)["model"] if eligible else None
    warnings = []
    for row in candidates:
        if row.get("rejected_flights"):
            warnings.append(
                row["model"] + " rejected: " + ", ".join(row["rejected_flights"])
            )
        if row.get("gap_warning"):
            warnings.append(row["model"] + ": TRAIN–validation gap >25 points")
        if (
            row.get("component_mean_fit") is not None
            and row["component_mean_fit"] < row["mean_fit"] - 20
        ):
            warnings.append(
                row["model"]
                + ": model DDV hatasına uyum sağlamış olabilir (Component >20 puan düşük)"
            )
    if stopped:
        warnings.append(stopped + "; completed candidates retained")
    if not selected:
        warnings.append("No eligible complete candidate; no model selected.")
    result = dict(
        id=out.name,
        selected=selected,
        config=dict(
            source=source,
            seed=seed,
            reliableSelection=True,
            selectionBasis=basis,
            include_test=False,
            maxCandidates=max_candidates,
            maxMinutes=max_minutes,
            clipRate=clip,
        ),
        summary=summarize(per, {r["model"] for r in candidates}),
        per_flight=per,
        search=candidates,
        candidate_flights=candidate_rows,
        diagnostics=[dict(warning=w) for w in warnings],
        split=[
            dict(flight=f.name, split=s, samples=len(f.time))
            for s, fs in groups.items()
            for f in fs
        ],
        dataset_sha256=fingerprint(flights),
        timings=dict(total=time.perf_counter() - start),
        validation_scores={
            r["model"]: r.get("mean_fit") for r in candidates if r["model"] in models
        },
        selection_policy="max minimum flight Fit then macro mean; reject negative/nonfinite; TEST closed",
        notice=(
            "Synthetic experiment; not real-flight validation. "
            if source != "user"
            else ""
        )
        + f"Selection: {basis}. TEST closed. "
        + (
            "Validation <3: automatic TRAIN-only LOFO. "
            if basis != "validation"
            else ""
        )
        + "; ".join(warnings),
        stopped=stopped,
    )
    (out / "models.json").write_text(json.dumps(serial(models)), encoding="utf8")
    (out / "flights.json").write_text(
        json.dumps([f.name for f in flights]), encoding="utf8"
    )
    np.savez_compressed(
        out / "flights.npz",
        **{f"f{i}": np.column_stack([f.time, f.values]) for i, f in enumerate(flights)},
    )
    (out / "result.json").write_text(
        json.dumps(serial(result), indent=2), encoding="utf8"
    )
    write_workbook(out, result)
    with pd.ExcelWriter(
        out / "comparison.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        pd.DataFrame(serial(candidate_rows)).to_excel(
            writer, sheet_name="CandidateFlights", index=False
        )
    log(f"Finished: {out}; selected={selected}")
    return serial(result), models, out
