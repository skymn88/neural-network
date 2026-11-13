"""Emit actuator_params.h from a saved models.json entry.
Usage: python gen_params.py models.json MODEL_NAME [out.h]"""
import json, sys

def arr(v):
    return "{" + ", ".join(repr(float(x)) for x in v) + "}"

def main():
    path, name = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "actuator_params.h"
    M = json.load(open(path))[name]
    d, r, a = M["ddv"], M["ram"], M["angle"]
    if r["kind"] != "null_rate":
        raise SystemExit("only ram.kind == null_rate is supported")
    if d.get("basis", "legacy") != "rich":
        raise SystemExit("only ddv.basis == rich is supported")
    o = r["observer"]
    mode = {"legacy": 0, "stationary": 1, "envelope": 2}[o.get("mode", "legacy")]
    zb = r.get("zero_band")
    L = [f"/* generated from {path}:{name} - do not edit */", "#ifndef ACTUATOR_PARAMS_H", "#define ACTUATOR_PARAMS_H",
         "#define FS 64.0",
         f"#define DDV_LAGS {d['lags']}", f"#define DDV_SCALE {float(d['scale'])!r}",
         f"static const double DDV_W[{len(d['weights'])}] = {arr(d['weights'])};",
         f"#define OBS_MODE {mode}", f"#define OBS_CENTER {float(o['center'])!r}",
         f"#define OBS_GATE0 {float(o['gate'][0])!r}", f"#define OBS_GATE1 {float(o['gate'][1])!r}",
         f"#define OBS_TAU {float(o['tau'])!r}", f"#define OBS_MAXSTEP {float(o['max_step'])!r}",
         f"#define OBS_ADAPTIVE {int(bool(o['adaptive']))}",
         f"#define OBS_QW {int(o.get('quiet_window', 1))}", f"#define OBS_QRANGE {float(o.get('quiet_range', 0))!r}",
         f"#define OBS_AGE {float(o.get('envelope_age', 0))!r}",
         f"#define RAM_DELAY {int(r['delay'])}",
         f"#define RAM_NK {len(r['knots'])}",
         f"static const double RAM_KNOTS[RAM_NK] = {arr(r['knots'])};",
         f"static const double RAM_VALUES[RAM_NK] = {arr(r['values'])};",
         f"#define RAM_LO {float(r['bounds'][0])!r}", f"#define RAM_HI {float(r['bounds'][1])!r}",
         f"#define RAM_HAS_ZB {int(zb is not None)}",
         f"#define RAM_ZB0 {float(zb[0]) if zb else 0.0!r}", f"#define RAM_ZB1 {float(zb[1]) if zb else 0.0!r}",
         f"#define ANG_N {len(a['weights'])}", f"#define ANG_SCALE {float(a['scale'])!r}",
         f"static const double ANG_W[ANG_N] = {arr(a['weights'])};", "#endif", ""]
    open(out, "w").write("\n".join(L))
    print("wrote", out)

main()
