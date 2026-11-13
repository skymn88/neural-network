"""Parity of the C core vs Python rollout. Usage: python run_parity.py models.json [model ...]"""
import json, subprocess, sys, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
from actuator.inference import rollout, predict_ddv, angle_values
from actuator.data import pulse_flights
path = sys.argv[1]; names = sys.argv[2:] or list(json.load(open(path)))
rng = np.random.default_rng(3)
t = np.arange(64 * 120)
cur = 3 * np.sin(2 * np.pi * t / 700) * np.sin(2 * np.pi * t / 3100) + 0.4 * rng.standard_normal(len(t)) * (t % 1500 < 700)
cur[2000:2400] = 0; cur[4000:6000] *= 0.01
for name in names:
    M = json.load(open(path))[name]
    try: subprocess.run([sys.executable, os.path.join(here, "gen_params.py"), path, name, os.path.join(here, "actuator_params.h")], check=True, capture_output=True)
    except subprocess.CalledProcessError: print(name, "skipped (unsupported)"); continue
    subprocess.run(["gcc", "-O2", "-I", here, "-o", "/tmp/parity", os.path.join(here, "parity_test.c"), "-lm"], check=True)
    ddv = predict_ddv(M["ddv"], cur); out = rollout(M, cur, 0.0)
    x = out[1]
    ang = angle_values(M["angle"], np.asarray(x))
    np.savetxt("/tmp/par.txt", np.c_[cur, ddv, x, ang], fmt="%.17g")
    print(name, subprocess.run(["/tmp/parity", "/tmp/par.txt", "0"], capture_output=True, text=True).stdout.strip())
