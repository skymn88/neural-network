import sys, time, json, copy, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from actuator.generalization import generalization_data, training_speed_limit, clipped_reference
from actuator.models import fit_ddv
from actuator.null_rate import fit_null_rate, refine_window_rate
from actuator.inference import rollout
from actuator.experiment import metrics
t0 = time.time()
old, groups, extra, newval = generalization_data()
train = groups["train"] + extra[:7]; val = groups["validation"] + newval
limit = training_speed_limit(train)
vals = np.concatenate([f.values for f in train])
ddv, _ = fit_ddv(train, train, length=16, basis="rich", fixed_ridge=1e-3)
scale = max(float(np.std(vals[:, 2])), 1e-9)
angle = dict(scale=scale, weights=np.polynomial.polynomial.polyfit(vals[:, 2] / scale, vals[:, 3], 3).tolist())
print("setup", round(time.time() - t0), flush=True)
def evaluate(m, flights):
    out = {}
    for stage in ("Cascade", "Component"):
        rows = []
        for f in flights:
            z = None if stage == "Cascade" else f.values[:, 1]
            x = rollout(m, f.values[:, 0], f.values[0, 2], z)[1]
            rows.append(metrics(f.values[32:, 2], x[32:]))
        sse = sum(r["sse"] for r in rows); sst = sum(r["sst"] for r in rows); fits = [r["fit"] for r in rows]
        out[stage] = dict(pooled=100 * (1 - (sse / sst) ** .5), macro=float(np.mean(fits)), min=float(min(fits)))
    return out
CFG = [
 ("stat_t2_ge1.5_win2", dict(observer="stationary", tau=2, gate_expansion=1.5), 2),
 ("stat_t2_ge1.5_nowin", dict(observer="stationary", tau=2, gate_expansion=1.5), 0),
 ("env_t2_ge1.5_win2", dict(observer="envelope", tau=2, gate_expansion=1.5), 2),
 ("env_t2_ge1.5_nowin", dict(observer="envelope", tau=2, gate_expansion=1.5), 0),
 ("stat_t1_ge1.5_win2", dict(observer="stationary", tau=1, gate_expansion=1.5), 2),
 ("stat_t2_ge1.0_win2", dict(observer="stationary", tau=2, gate_expansion=1.0), 2),
 ("stat_t2_q1s_win2", dict(observer="stationary", tau=2, gate_expansion=1.5, quiet_seconds=1.0), 2),
 ("stat_t4_ge1.5_win2", dict(observer="stationary", tau=4, gate_expansion=1.5), 2),
 ("stat_t8_ge1.5_win2", dict(observer="stationary", tau=8, gate_expansion=1.5), 2),
 ("stat_t16_ge1.5_win2", dict(observer="stationary", tau=16, gate_expansion=1.5), 2),
 ("stat_t8_ge1.5_win5", dict(observer="stationary", tau=8, gate_expansion=1.5), 5),
]
res = json.load(open("audit_results/null_rate_tau_grid_20261116/result.json"))
for name, kw, win in CFG:
    if name in res: continue
    t = time.time()
    try:
        m = fit_null_rate(train, ddv, angle, adaptive=True, knots=81, ridge=1e-3, gate_source="predicted", **kw)
        if win: m = refine_window_rate(m, train, seconds=win, ridge=1e-2)
        m = clipped_reference(m, limit)
        res[name] = dict(train=evaluate(m, train), val=evaluate(m, val), seconds=time.time() - t)
        json.dump({"model": m} if False else res, open("audit_results/null_rate_tau_grid_20261116/result.json", "w"), indent=1)
        json.dump(m, open("model_%s.json" % name, "w"))
        r = res[name]
        print("%-22s Casc tr %.1f val %.1f(min %.1f) | Comp tr %.1f val %.1f(min %.1f)  %ds" % (name, r["train"]["Cascade"]["pooled"], r["val"]["Cascade"]["pooled"], r["val"]["Cascade"]["min"], r["train"]["Component"]["pooled"], r["val"]["Component"]["pooled"], r["val"]["Component"]["min"], r["seconds"]), flush=True)
    except Exception as e:
        print(name, "FAILED", repr(e)[:120], flush=True)
print("total", round(time.time() - t0))
