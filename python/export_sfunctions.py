"""Embed a selected JSON LSTM/GRU rate model into three portable C MEX blocks."""

import argparse, hashlib, json, math, shutil
from pathlib import Path
import numpy as np

TEMPLATES = Path(__file__).parent / "actuator/c_templates"


def number(value):
    value = float(value)
    if math.isnan(value):
        raise ValueError("NaN coefficient")
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return format(value, ".17g")


def export(path, selected, out):
    path = Path(path)
    out = Path(out)
    catalog = json.loads(path.read_text())
    model = catalog[selected] if selected else catalog
    ram = model["ram"]
    members = ram["members"] if ram["kind"] == "rate_ensemble" else [ram]
    if ram["kind"] not in ("rate_ensemble", "sequence_rate"):
        raise ValueError(
            "Export supports LSTM/GRU sequence_rate and rate_ensemble only"
        )
    ref = members[0]["reference"]
    obs = ref["observer"]
    ddv = model["ddv"]
    angle = model["angle"]
    if obs.get("mode") != "stationary":
        raise ValueError("Export currently requires stationary null observer")
    for member in members:
        if member["kind"] != "sequence_rate" or member["sequence"]["family"] not in (
            "lstm",
            "gru",
        ):
            raise ValueError("LSTM/GRU members required")
        if member["reference"] != ref or member["bounds"] != members[0]["bounds"]:
            raise ValueError("Shared reference and bounds required")
    if ddv.get("basis", "legacy") not in ("rich", "legacy"):
        raise ValueError("Unsupported DDV basis")
    if any(float(v) <= 0 for v in [ddv["scale"], angle["scale"], obs["tau"]]):
        raise ValueError("Positive scales required")
    h = max(m["sequence"]["hidden"] for m in members)
    count = len(members)
    defines = dict(
        DDV_HISTORY=ddv["lags"] + 1,
        DDV_FEATURES=len(ddv["weights"]),
        DDV_RICH=int(ddv.get("basis", "legacy") == "rich"),
        MAX_HIDDEN=h,
        MEMBER_COUNT=count,
        QUIET_WINDOW=obs["quiet_window"],
        REF_DELAY=ref["delay"],
        REF_KNOTS=len(ref["knots"]),
        ANGLE_COUNT=len(angle["weights"]),
    )
    if defines["QUIET_WINDOW"] < 1 or defines["REF_DELAY"] < 0:
        raise ValueError("Invalid observer state sizes")
    expected = 1 + defines["DDV_HISTORY"] * (9 if defines["DDV_RICH"] else 4)
    if defines["DDV_FEATURES"] != expected:
        raise ValueError("DDV coefficient shape mismatch")
    header = (
        """#ifndef ACTUATOR_CORE_H
#define ACTUATOR_CORE_H
#include <stddef.h>
"""
        + "".join(f"#define {k} {v}\n" for k, v in defines.items())
        + """
typedef struct {double ready,u[DDV_HISTORY];} BBDdvState;
typedef struct {double ready,x,center,previous,was_quiet;double ddv[QUIET_WINDOW],delay[REF_DELAY+1];double hidden[MEMBER_COUNT][MAX_HIDDEN],cell[MEMBER_COUNT][MAX_HIDDEN];} BBRamState;
size_t bb_ddv_state_size(void);size_t bb_ram_state_size(void);
void bb_ddv_init(BBDdvState *s);double bb_ddv_step(BBDdvState *s,double current);
void bb_ram_init(BBRamState *s,double x0);double bb_ram_step(BBRamState *s,double ddv);
double bb_angle(double ram);
#endif
"""
    )
    coefficients = [
        "/* Exported from models.json; do not edit weights by hand. */",
        "#include <math.h>",
    ]

    def scalar(name, value):
        coefficients.append(f"#define {name} ({number(value)})")

    def array(name, value):
        a = np.asarray(value, float)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite array " + name)
        coefficients.append(
            f"static const double {name}[{a.size}]={{"
            + ",".join(number(v) for v in a.ravel())
            + "};"
        )

    scalar("DDV_SCALE", ddv["scale"])
    array("DDV_WEIGHTS", ddv["weights"])
    if defines["DDV_RICH"]:
        array("DDV_TANH", [0.5, 1, 2, 4, 8])
        array("DDV_DEAD", [0.05, 0.15])
    scalar("ANGLE_SCALE", angle["scale"])
    array("ANGLE_COEFF", angle["weights"])
    for name, value in dict(
        REF_CENTER=obs["center"],
        REF_GATE_LO=obs["gate"][0],
        REF_GATE_HI=obs["gate"][1],
        REF_QUIET_RANGE=obs["quiet_range"],
        REF_ALPHA=1 - np.exp(-1 / (64 * obs["tau"])),
        REF_ADAPTIVE=int(obs["adaptive"]),
        REF_MAP_LIMIT=ref.get("map_rate_limit", float("inf")),
        REF_ZERO_BAND=int("zero_band" in ref),
        REF_ZERO_LO=ref.get("zero_band", [0, 0])[0],
        REF_ZERO_HI=ref.get("zero_band", [0, 0])[1],
        RAM_LO=members[0]["bounds"][0],
        RAM_HI=members[0]["bounds"][1],
    ).items():
        scalar(name, value)
    array("REF_U", ref["knots"])
    array("REF_RATE", ref["values"])
    coefficients.append(
        "typedef struct {int hidden,gates,residual,gate;double rate_scale,correction_limit,output_limit,bias;const double *mean,*scale,*wi,*wh,*bi,*bh,*head;} BBWeights;"
    )
    entries = []
    for i, member in enumerate(members):
        m = member["sequence"]
        size = m["hidden"]
        gates = 4 if m["family"] == "lstm" else 3
        shapes = dict(
            mean=(5,),
            scale=(5,),
            wi=(gates * size, 5),
            wh=(gates * size, size),
            bi=(gates * size,),
            bh=(gates * size,),
            head=(size,),
        )
        for key, shape in shapes.items():
            if np.asarray(m[key]).shape != shape:
                raise ValueError("Sequence shape mismatch: " + key)
            array(f"M{i}_{key}", m[key])
        if any(v <= 0 for v in m["scale"]):
            raise ValueError("Positive feature scale required")
        values = (
            [str(size), str(gates), str(int(m["residual"])), str(int(m["gate"]))]
            + [
                number(v)
                for v in (
                    m["rate_scale"],
                    m.get("correction_limit", float("inf")),
                    m.get("output_rate_limit", m["rate_limit"]),
                    m["bias"],
                )
            ]
            + [f"M{i}_{key}" for key in shapes]
        )
        entries.append("{" + ",".join(values) + "}")
    coefficients.append(
        "static const BBWeights MEMBERS[MEMBER_COUNT]={" + ",\n".join(entries) + "};"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "actuator_core.h").write_text(header)
    (out / "model_coefficients.h").write_text("\n".join(coefficients) + "\n")
    shutil.copy2(TEMPLATES / "actuator_core.c", out / "actuator_core.c")
    template = (TEMPLATES / "sfunction.c").read_text()
    for name, state, init, step in [
        (
            "bb_curr_ddv",
            "BBDdvState",
            "bb_ddv_init((BBDdvState*)ssGetDWork(S,0));",
            "bb_ddv_step",
        ),
        (
            "bb_ddv_ram",
            "BBRamState",
            "bb_ram_init((BBRamState*)ssGetDWork(S,0),mxGetScalar(ssGetSFcnParam(S,0)));",
            "bb_ram_step",
        ),
        ("bb_ram_angle", None, "(void)S;", "bb_angle"),
    ]:
        params = name == "bb_ddv_ram"
        fields = dict(
            NAME=name,
            NPARAMS=int(params),
            NDWORK=int(state is not None),
            INIT=init,
            CHECK_PARAMETERS="",
            VALIDATE="",
            DWORK="",
            OUTPUT="    y[0]=bb_angle(u[0]);",
            UPDATE="    (void)S;",
        )
        if params:
            fields["CHECK_PARAMETERS"] = """#define MDL_CHECK_PARAMETERS
static void mdlCheckParameters(SimStruct *S){const mxArray *p=ssGetSFcnParam(S,0);if(!mxIsDouble(p)||mxIsComplex(p)||mxIsSparse(p)||mxGetNumberOfElements(p)!=1||!mxIsFinite(mxGetScalar(p)))ssSetErrorStatus(S,"RAM x0 must be one finite real double in physical units.");}"""
            fields["VALIDATE"] = (
                "    mdlCheckParameters(S);if(ssGetErrorStatus(S)!=NULL)return;ssSetSFcnParamTunable(S,0,0);"
            )
        if state:
            fields["DWORK"] = (
                f"    ssSetDWorkWidth(S,0,(int_T)(sizeof({state})/sizeof(real_T)));ssSetDWorkDataType(S,0,SS_DOUBLE);ssSetDWorkUsedAsDState(S,0,1);"
            )
            fields["OUTPUT"] = (
                f"    {state} next;memcpy(&next,ssGetDWork(S,0),sizeof(next));y[0]={step}(&next,u[0]);"
            )
            fields["UPDATE"] = (
                f"    const real_T *u=(const real_T*)ssGetInputPortSignal(S,0);(void){step}(({state}*)ssGetDWork(S,0),u[0]);"
            )
        text = template
        for key, value in fields.items():
            text = text.replace("@" + key + "@", str(value))
        (out / (name + ".c")).write_text(text)
    shutil.copy2(TEMPLATES / "build_sfunctions.m", out / "build_sfunctions.m")
    shutil.copy2(TEMPLATES / "verify_sfunctions.m", out / "verify_sfunctions.m")
    from actuator.inference import rollout

    current = np.r_[
        np.repeat(0.4, 64),
        np.repeat(-0.15, 96),
        np.zeros(160),
        np.repeat(0.1, 96),
        np.zeros(160),
        0.1 * np.sin(np.arange(192) / 12),
    ]
    predicted = np.array(rollout(model, current, -80.0)).T
    np.savetxt(
        out / "parity_fixture.csv",
        np.column_stack([np.arange(len(current)) / 64, current, predicted]),
        delimiter=",",
        header="time,current,ddv,ram,angle",
        comments="",
        fmt="%.17g",
    )
    metadata = dict(
        model=selected,
        models_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selected_model_sha256=hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        sample_rate=64,
        training_seeds=model.get("training_seeds"),
        members=count,
        families=[m["sequence"]["family"] for m in members],
        ram_initialization="physical RAM x0 block parameter; first output equals x0",
        mexw64_built=False,
    )
    (out / "export_manifest.json").write_text(json.dumps(metadata, indent=2))
    (out / "readme.md").write_text("""# Generated C MEX S-Functions

Coefficients are embedded in `model_coefficients.h`, copied from the selected JSON model. No JSON/Python/PyTorch is required at inference time. Keep the core/header/coefficient files next to the three block sources.

On **Windows x64 MATLAB R2019a with Simulink and a supported C compiler**, change MATLAB Current Folder to this folder and run:

```matlab
build_sfunctions
verify_sfunctions
```

`verify_sfunctions` creates a temporary three-block model and checks its output against the shipped deterministic Python fixture; it then closes that temporary model. This MATLAB test has not been run in the Linux environment.

The first use may require `mex -setup C`. The command produces `bb_curr_ddv.mexw64`, `bb_ddv_ram.mexw64`, `bb_ram_angle.mexw64`. They are not precompiled in the Linux export environment.

Add three **S-Function** blocks with names `bb_curr_ddv`, `bb_ddv_ram`, `bb_ram_angle`; wire Current → DDV → RAM → Angle. The RAM block has **one parameter: initial measured RAM position**, in physical units (for example `-100`). Other blocks have no parameters. All ports are scalar real double and all outputs use original physical units. These are Level-2 C MEX S-Functions, not MATLAB Level-2 functions.

All three model blocks run at **1/64 s**. Do not change to 0.005 s: doing so changes learned dynamics. If the surrounding interface needs 200 Hz, use a Rate Transition/zero-order-hold boundary; it does not create new 200 Hz learned dynamics and its scheduling delay must be checked in the host model. No interpolation/look-ahead is used in the exported model.

Hidden/cell/null/delay/position state is per block instance in DWork; repeated mdlOutputs calls do not advance state. mdlUpdate commits one 64 Hz step; initialize/reset starts a new flight. No measured RAM is fed back after initial x0. This is a synthetic validation-selected research model, not real-flight qualification. C core parity is checked in Linux; MATLAB/Simulink execution and Windows MEX loading must be checked locally.
""")
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models_json")
    p.add_argument("--model")
    p.add_argument("--output", default="sfunctions/selected")
    args = p.parse_args()
    print(json.dumps(export(args.models_json, args.model, args.output), indent=2))
