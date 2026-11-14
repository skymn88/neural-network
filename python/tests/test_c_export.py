import ctypes as c
import json
import shutil
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from actuator.data import ROOT
from actuator.inference import rollout
from actuator.sequence_train import prepare_sequences, fit_sequence
from actuator.generalization import make_rate_ensemble
from export_sfunctions import export


def library(folder):
    target = folder / "libactuator.so"
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            str(folder / "actuator_core.c"),
            "-o",
            str(target),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(target))
    for name in ("ddv", "ram"):
        getattr(lib, "bb_" + name + "_state_size").restype = c.c_size_t
        getattr(lib, "bb_" + name + "_step").argtypes = [c.c_void_p, c.c_double]
        getattr(lib, "bb_" + name + "_step").restype = c.c_double
    lib.bb_ddv_init.argtypes = [c.c_void_p]
    lib.bb_ram_init.argtypes = [c.c_void_p, c.c_double]
    lib.bb_angle.argtypes = [c.c_double]
    lib.bb_angle.restype = c.c_double
    return lib


def c_rollout(lib, current, x0, override=None):
    d = c.create_string_buffer(lib.bb_ddv_state_size())
    r = c.create_string_buffer(lib.bb_ram_state_size())
    lib.bb_ddv_init(d)
    lib.bb_ram_init(r, x0)
    out = np.empty((len(current), 3))
    for i, u in enumerate(current):
        z = lib.bb_ddv_step(d, u) if override is None else override[i]
        x = lib.bb_ram_step(r, z)
        out[i] = [z, x, lib.bb_angle(x)]
    return out


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("gcc"), "Linux GCC required; Windows uses verify_sfunctions in MATLAB")
class ExportTests(unittest.TestCase):
    def test_lstm_portable_and_sfunction_callbacks(self):
        source = ROOT / "python/reference_results/20261113_051726_000000/models.json"
        model = json.loads(source.read_text())["n14_q999"]
        u = np.r_[
            np.repeat(0.4, 80),
            np.repeat(-0.15, 100),
            np.zeros(300),
            np.random.default_rng(42).normal(0, 0.1, 900),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export(source, "n14_q999", out)
            lib = library(out)
            expected = np.array(rollout(model, u, -80)).T
            np.testing.assert_allclose(
                c_rollout(lib, u, -80), expected, atol=1e-8, rtol=1e-10
            )
            np.testing.assert_allclose(
                c_rollout(lib, u, -80, override=u),
                np.array(rollout(model, u, -80, ddv_override=u)).T,
                atol=1e-8,
                rtol=1e-10,
            )
            # Reinitialization reproduces the same first flight exactly.
            np.testing.assert_array_equal(
                c_rollout(lib, u, -80), c_rollout(lib, u, -80)
            )
            for block, state in [
                ("bb_curr_ddv", "ddv"),
                ("bb_ddv_ram", "ram"),
                ("bb_ram_angle", "angle"),
            ]:
                driver = out / "driver.c"
                driver.write_text(
                    '''#define MATLAB_MEX_FILE
#include <assert.h>
#include <string.h>
#include "'''
                    + block
                    + """.c"
int main(void){SimStruct s,t;int k;double first,copy[8192];
memset(&s,0,sizeof(s));s.provided="""
                    + str(int(state == "ram"))
                    + """;s.param.count=1;s.param.type=1;s.param.value=-80;
mdlInitializeSizes(&s);assert(!s.error);mdlInitializeSampleTimes(&s);assert(s.ts==1.0/64.0);mdlInitializeConditions(&s);t=s;
for(k=0;k<30;k++){s.u=t.u=0.1*sin(k);memcpy(copy,s.work,sizeof(copy));mdlOutputs(&s,0);first=s.y;mdlOutputs(&s,0);assert(s.y==first);assert(memcmp(copy,s.work,sizeof(copy))==0);mdlOutputs(&t,0);assert(t.y==s.y);mdlUpdate(&s,0);mdlUpdate(&t,0);assert(memcmp(s.work,t.work,sizeof(s.work))==0);}
mdlInitializeConditions(&s);s.u=.1;mdlOutputs(&s,0);first=s.y;mdlInitializeConditions(&s);mdlOutputs(&s,0);assert(first==s.y);mdlTerminate(&s);return 0;}
"""
                )
                subprocess.run(
                    [
                        "gcc",
                        "-std=c99",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-I" + str(ROOT / "python/tests/cstub"),
                        "-I" + str(out),
                        str(driver),
                        str(out / "actuator_core.c"),
                        "-lm",
                        "-o",
                        str(out / "driver"),
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run([str(out / "driver")], check=True, capture_output=True)

    def test_gru_and_unsupported_observer(self):
        source = ROOT / "python/reference_results/20261113_051726_000000/models.json"
        m = json.loads(source.read_text())["n14_q999"]
        ref = dict(
            ddv=m["ddv"], angle=m["angle"], ram=m["ram"]["members"][0]["reference"]
        )
        # Tiny training fixture checks the actual GRU gate ordering in the C exporter.
        from actuator.data import Flight

        t = np.arange(512) / 64
        u = 0.1 * np.sin(t)
        x = np.cumsum(u) / 64
        f = Flight("fixture", t, np.column_stack([u, u, x, -x]))
        trained = fit_sequence(
            prepare_sequences([f], ref),
            ref,
            family="gru",
            hidden=4,
            steps=2,
            horizon=96,
        )
        ensemble = make_rate_ensemble([trained, trained], [42, 43])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p = out / "model.json"
            p.write_text(json.dumps(ensemble))
            export(p, None, out)
            lib = library(out)
            np.testing.assert_allclose(
                c_rollout(lib, u, -80),
                np.array(rollout(ensemble, u, -80)).T,
                atol=1e-8,
                rtol=1e-10,
            )
            ensemble["ram"]["members"][0]["reference"]["observer"]["mode"] = "envelope"
            p.write_text(json.dumps(ensemble))
            with self.assertRaisesRegex(ValueError, "stationary"):
                export(p, None, out)
