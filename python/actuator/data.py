from dataclasses import dataclass
from typing import Optional
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
    truth: Optional[dict] = None  # simulator-only audit; learners must never read this

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
    seed=123, durations=None, profile="pulse", calibration=None, valve_variation=None
):
    """Separate hidden simulator. Learners never receive its parameters/state.
    Irregular signed pulses + long holds, noisy current, lowpass DDV, asymmetric
    bounded velocity and exact position retention at neutral. No vibration yet.
    """
    if valve_variation is not None:
        if profile != "flightLike":
            raise ValueError("valve_variation requires flightLike")
        allowed = {"gain_sd", "bias_sd", "drift_amp"}
        if not isinstance(valve_variation, dict) or set(valve_variation) - allowed:
            raise ValueError("Unknown valve_variation setting")
        if any(not np.isfinite(v) or v < 0 for v in valve_variation.values()):
            raise ValueError("Valve variation values must be finite and nonnegative")
    if durations is None:
        durations = (
            FLIGHT_LIKE_DURATIONS
            if profile == "flightLike"
            else (500, 550, 600, 650, 700, 750, 800, 900, 1000, 1100, 850)
        )
    if profile == "flightLike":
        return _flight_like(seed, durations, calibration, valve_variation)
    if profile != "pulse":
        raise ValueError("Unknown synthetic profile")
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


FLIGHT_LIKE_DURATIONS = (
    2200,
    2400,
    2100,
    2800,
    2300,
    2600,
    2500,
    2700,
    2250,
    2450,
    2350,
)


def _flight_like(seed, durations, calibration=None, valve_variation=None):
    """Opt-in extension of pulse_flights; Truth is excluded from all learners.
    A neutral valve cannot lift RAM from its stop: explicit opposite pulses are
    used for release. Slow start uses a near-neutral command before parking,
    rather than inventing slow motion under a large fully-open spool command.
    """
    from scipy.signal import lfilter

    c = calibration or {}
    flights = []
    for j, duration in enumerate(durations):
        rng = np.random.default_rng(seed + 101 * j)
        n = int(duration * FS)
        if duration < 900:
            raise ValueError("flightLike needs >=900 seconds for park and long holds")
        t = np.arange(n) / FS
        base = rng.uniform(*c.get("null_range", [-0.05, 0.03]))
        drift_amp = (
            rng.uniform(0.025, 0.05) if not calibration else c["drift_amplitude"]
        )
        null = base + drift_amp * np.sin(
            2 * np.pi * t / rng.uniform(1100, 2400) + rng.uniform(-3, 3)
        )
        band = rng.uniform(0.012, 0.024) if not calibration else c["deadband"]
        vmax = c.get("max_rate", 55.0) * rng.uniform(0.995, 1.005)
        lo, hi = c.get("ram_bounds", [-100.0, 100.0])
        park = c.get("park_ddv", 0.4)
        excitation = c.get("excitation", 0.4)
        relative = np.zeros(n)
        phase = np.full(n, "hold", dtype="<U12")
        events = []

        def segment(start, end, value, label):
            a = max(0, int(start * FS))
            b = min(n, int(end * FS))
            relative[a:b] = value
            phase[a:b] = label
            events.append({"start": a / FS, "end": b / FS, "kind": label})

        park_end = min(rng.uniform(350, 600), duration * 0.25)
        # Positive spool closes towards the lower stop (negative RAM direction).
        segment(0, park_end, 0, "park")
        relative[: int(park_end * FS)] = park - null[: int(park_end * FS)]
        initial = lo
        if j % 3 == 0:
            initial = lo + rng.uniform(0.05, 0.2) * (hi - lo)
            slow_end = rng.uniform(50, 100)
            segment(0, slow_end, band + 0.00045, "slow_start")
        release = ((lo + hi) / 2 - lo) / vmax
        segment(
            park_end,
            park_end + release,
            c.get("rate_direction", -1) * excitation,
            "release",
        )
        k = park_end + release + min(rng.uniform(120, 220), duration * 0.1)
        cutoff = duration - rng.uniform(70, 110)
        # Mix long level changes, symmetric/asymmetric pulse pairs and bursts.
        group = 0
        while k < cutoff - 60:
            if group == 0 and cutoff - k > 650:
                segment(k, k + 12, excitation, "rectangular")
                k += 12
                hold = min(rng.uniform(500, 620), cutoff - k - 35)
                segment(k, k + hold, 0, "long_hold")
                k += hold
                segment(
                    k, k + release, c.get("rate_direction", -1) * excitation, "release"
                )
                k += release
            else:
                kind = group % 3
                sign = float(rng.choice([-1, 1]))
                if kind == 0:
                    width = rng.uniform(5, 20)
                    segment(
                        k,
                        k + width,
                        sign * excitation * rng.uniform(0.65, 1.15),
                        "rectangular",
                    )
                    k += width
                    segment(
                        k,
                        k + width,
                        -sign * excitation * rng.uniform(0.65, 1.15),
                        "rectangular",
                    )
                    k += width
                elif kind == 1:
                    width = rng.uniform(8, 22)
                    a = int(k * FS)
                    b = min(n, int((k + width) * FS))
                    q = np.linspace(0, 1, b - a)
                    relative[a:b] = (
                        sign
                        * excitation
                        * np.minimum(4 * q, np.minimum(1, 4 * (1 - q)))
                    )
                    phase[a:b] = "trapezoid"
                    events.append({"start": k, "end": k + width, "kind": "trapezoid"})
                    k += width
                    segment(k, k + width, -sign * excitation * 0.65, "return")
                    k += width
                else:
                    width = rng.uniform(8, 90)
                    a = int(k * FS)
                    b = min(n, int((k + width) * FS))
                    q = np.arange(b - a) / FS
                    freq = rng.uniform(0.15, 1.5)
                    relative[a:b] = (
                        excitation
                        * rng.uniform(0.5, 1)
                        * np.sin(2 * np.pi * freq * q)
                        * np.minimum(q, 1)
                        * np.minimum(width - q, 1)
                    )
                    phase[a:b] = "burst"
                    events.append({"start": k, "end": k + width, "kind": "burst"})
                    k += width
            hold = rng.uniform(20, 60)
            segment(k, k + hold, 0, "hold")
            k += hold
            group += 1
        segment(cutoff, duration, 0, "park")
        relative[int(cutoff * FS) :] = park - null[int(cutoff * FS) :]
        command = null + relative
        gain = c.get("valve_gain", 1.0)
        bias = c.get("valve_bias", 0.0)
        if valve_variation:
            # Opt-in: current->DDV offset/gain differ per flight and drift slowly. The
            # governing equations are unchanged; gain also scales input noise slightly.
            vr = np.random.default_rng(seed + 101 * j + 7919)
            gain *= 1 + valve_variation.get("gain_sd", 0.0) * vr.standard_normal()
            bias = (
                bias
                + valve_variation.get("bias_sd", 0.0) * vr.standard_normal()
                + valve_variation.get("drift_amp", 0.0)
                * np.sin(2 * np.pi * t / vr.uniform(900, 2400) + vr.uniform(-3, 3))
            )
        if not np.isfinite(gain) or gain <= 1e-6:
            raise ValueError("Sampled valve gain must be positive; reduce gain_sd")
        current = (command - bias) / gain + rng.normal(0, 0.0015, n)
        # Common physical dynamics, small flight-to-flight variation only.
        a = np.exp(-1 / (FS * 0.055))
        valve_input = gain * current + bias
        z = lfilter([1 - a], [1, -a], valve_input, zi=[a * valve_input[0]])[0]
        deviation = np.roll(z, 2) - null
        deviation[:2] = z[0] - null[:2]
        rate = (
            c.get("rate_direction", -1)
            * vmax
            * np.tanh(np.sign(deviation) * np.maximum(abs(deviation) - band, 0) / 0.055)
        )
        # Exact retention in the true neutral band; no leakage term.
        x = np.empty(n)
        x[0] = initial
        for k in range(1, n):
            x[k] = min(hi, max(lo, x[k - 1] + rate[k] / FS))
        measured_ddv = np.r_[z[0], z[:-1]] + rng.normal(0, 0.0008, n)
        measured_ram = x + rng.normal(0, 0.002, n)
        if calibration:
            angle = np.polynomial.polynomial.polyval(x, c["angle_weights"])
        else:
            angle = 2 - 0.29 * x + 0.0002 * x * x
        angle += rng.normal(0, 0.015, n)
        truth = {
            "null": null,
            "deadband": band,
            "base_null": base,
            "drift_amplitude": drift_amp,
            "ram_clean": x,
            "ddv_clean": z,
            "rate_requested": rate,
            "max_rate": vmax,
            "phase": phase,
            "events": events,
            "seed": seed + 101 * j,
            "synthetic": True,
            "profile": "flightLike",
            "valve_gain": float(gain),
            "valve_bias_mean": float(np.mean(bias)),
            "valve_bias": np.broadcast_to(bias, t.shape).copy(),
            "ram_unit": "user-like arbitrary unit; not verified mm",
        }
        flights.append(
            Flight(
                f"FLIKE_{j + 1:02}",
                t,
                np.c_[current, measured_ddv, measured_ram, angle],
                truth,
            )
        )
    return flights, split(flights, seed=42)
