from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
import hashlib
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
FS = 64
NAMES = ("CoilCurrent", "DDV", "RAM", "Deflection")


@dataclass
class Flight:
    name: str
    time: np.ndarray
    values: np.ndarray  # current, DDV, RAM, angle, physical units

    def slice(self, start, stop, suffix=""):
        return Flight(
            self.name + suffix,
            self.time[start:stop].copy(),
            self.values[start:stop].copy(),
        )


def align(name, signals, fs=FS):
    if len(signals) != 4:
        raise ValueError("Exactly four signal/time pairs are required")
    clean = []
    for t, x in signals:
        t, x = np.asarray(t, float).ravel(), np.asarray(x, float).ravel()
        ok = np.isfinite(t) & np.isfinite(x)
        t, x = t[ok], x[ok]
        order = np.argsort(t)
        t, x = t[order], x[order]
        t, ix = np.unique(t, return_index=True)
        x = x[ix]
        if len(t) < 10:
            raise ValueError(f"{name}: too few finite signal samples")
        dt = np.median(np.diff(t))
        if np.max(np.diff(t)) > 5 * dt:
            raise ValueError(f"{name}: signal contains a long time gap")
        native = 1 / dt
        if native > fs * 1.05:
            uniform = np.arange(t[0], t[-1] + dt * 0.1, dt)
            sampled = np.interp(uniform, t, x)
            ratio = Fraction(fs / native).limit_denominator(10000)
            x = resample_poly(
                sampled, ratio.numerator, ratio.denominator, padtype="line"
            )
            t = uniform[0] + np.arange(len(x)) / (
                native * ratio.numerator / ratio.denominator
            )
        clean.append((t, x))
    lo = max(t[0] for t, x in clean)
    hi = min(t[-1] for t, x in clean)
    time = lo + np.arange(int(np.floor((hi - lo) * fs)) + 1) / fs
    if len(time) < 128:
        raise ValueError(f"{name}: common time range too short")
    return Flight(
        name, time, np.column_stack([np.interp(time, t, x) for t, x in clean])
    )


def bundled():
    d = loadmat(ROOT / "data/synthetic/synthetic_rich.mat", simplify_cells=True)
    flights = [
        align(r["name"], [(r[k]["t"], r[k]["x"]) for k in NAMES]) for r in d["Raw"]
    ]
    ids = {
        s: [int(i) - 1 for i in np.atleast_1d(d[k])]
        for s, k in [
            ("train", "trainIDs"),
            ("validation", "validationIDs"),
            ("test", "testIDs"),
        ]
    }
    return flights, {s: [flights[i] for i in ix] for s, ix in ids.items()}


def split(flights, seed=42, assignments=None):
    if assignments:
        if any(a not in ("train", "validation", "test") for a in assignments):
            raise ValueError("Split labels must be train, validation or test")
        groups = {
            s: [f for f, a in zip(flights, assignments) if a == s]
            for s in ("train", "validation", "test")
        }
        if len(assignments) != len(flights) or any(not v for v in groups.values()):
            raise ValueError("Explicit split needs at least one flight in every group")
        return groups
    n = len(flights)
    if not n:
        raise ValueError("At least one flight is required")
    if n < 4:
        groups = {s: [] for s in ("train", "validation", "test")}
        for f in flights:
            a = int(0.7 * len(f.time))
            b = int(0.9 * len(f.time))
            guard = 64
            for s, lo, hi in [
                ("train", 0, a - guard),
                ("validation", a + guard, b - guard),
                ("test", b + guard, len(f.time)),
            ]:
                if hi - lo < 128:
                    raise ValueError("Flight too short for chronological guarded split")
                groups[s].append(f.slice(lo, hi, "_" + s))
        return groups
    if n < 10:
        counts = (n - 2, 1, 1)
    elif n == 10:
        counts = (7, 2, 1)
    elif n == 11:
        counts = (7, 2, 2)
    else:
        train = int(np.floor(0.7 * n))
        val = max(1, int(np.floor(0.2 * n)))
        counts = (train, val, n - train - val)
    ix = np.random.default_rng(seed).permutation(n)
    a, b = counts[0], sum(counts[:2])
    return {
        s: [flights[i] for i in ids]
        for s, ids in [("train", ix[:a]), ("validation", ix[a:b]), ("test", ix[b:])]
    }


def fingerprint(flights):
    h = hashlib.sha256()
    for f in flights:
        h.update(f.name.encode())
        h.update(np.asarray(f.time, dtype="<f8").tobytes())
        h.update(np.asarray(f.values, dtype="<f8").tobytes())
    return h.hexdigest()


def pulse_flights(
    seed=123, durations=(500, 550, 600, 650, 700, 750, 800, 900, 1000, 1100, 850)
):
    """Separate hidden simulator. Learners never receive its parameters/state.
    Irregular signed pulses + long holds, noisy current, lowpass DDV, asymmetric
    bounded velocity and exact position retention at neutral. No vibration yet.
    """
    flights = []
    for j, duration in enumerate(durations):
        rng = np.random.default_rng(seed + 101 * j)
        n = int(duration * FS)
        u = np.zeros(n)
        k = FS
        while k < n:
            width = int(rng.uniform(0.2, 4) * FS)
            hold = int(rng.uniform(2, 12) * FS)
            amp = rng.uniform(0.15, 1.2) * rng.choice([-1, 1])
            u[k : min(n, k + width)] = amp
            k += width + hold
        measured = u + rng.normal(0, 0.025, n)
        z = 0.0
        v = 0.0
        x = rng.uniform(-8, 8)
        rows = np.zeros((n, 4))
        for k in range(n):
            z += 0.18 * (np.tanh(measured[k]) - z)
            target = np.sign(z) * max(abs(z) - 0.06, 0) * (12 if z >= 0 else 10)
            v += 0.22 * (target - v)
            if abs(z) < 0.025 and abs(v) < 0.01:
                v = 0.0
            next_x = x + v / FS
            x = np.clip(next_x, -80, 80)
            if x != next_x:
                v = 0.0
            rows[k] = [measured[k], z, x, 0.35 * x + 0.000002 * x**3]
        flights.append(Flight(f"PULSE_{j + 1:02}", np.arange(n) / FS, rows))
    return flights, split(flights, seed=42)
