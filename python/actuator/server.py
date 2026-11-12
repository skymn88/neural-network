"""Local-only UI/API. Uploaded flight data never leaves this process/machine."""

import base64, io, json, threading, uuid
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .data import ROOT, NAMES, Flight, align, bundled, pulse_flights, split, fingerprint
from .experiment import run, serial
from .models import rollout

app = FastAPI(title="Actuator Lab")
jobs = {}
datasets = {}
lock = threading.Lock()
WEB = ROOT / "python/web"
RESULTS = ROOT / "python/results"
REFERENCES = ROOT / "python/reference_results"


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"detail": "Only same-origin requests are allowed"}, status_code=403
        )
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


class Upload(BaseModel):
    name: str
    content: str = Field(max_length=45000000)


class Inspect(BaseModel):
    file: Upload
    separator: Literal[",", ";", "\t"] = ","


def read_csv(file, sep):
    try:
        data = base64.b64decode(file.content, validate=True)
        if len(data) > 32 * 1024**2:
            raise ValueError("32 MiB file limit")
        return pd.read_csv(io.BytesIO(data), sep=sep)
    except Exception as e:
        raise HTTPException(400, f"CSV could not be read: {e}")


@app.post("/api/inspect")
def inspect(body: Inspect):
    d = read_csv(body.file, body.separator)
    return {"columns": list(d.columns), "rows": len(d)}


class Import(BaseModel):
    files: list[Upload] = Field(min_length=1, max_length=40)
    mapping: dict[str, str]
    times: dict[str, str]
    separator: Literal[",", ";", "\t"] = ","
    assignments: Optional[list[str]] = None


@app.post("/api/import")
def import_data(body: Import):
    if sum(len(f.content) for f in body.files) > 256 * 1024**2:
        raise HTTPException(400, "Combined upload exceeds 256 MiB encoded")
    if len({Path(f.name).stem for f in body.files}) != len(body.files):
        raise HTTPException(400, "Flight filenames must be unique")
    flights = []
    try:
        for file in body.files:
            frame = read_csv(file, body.separator)
            signals = [
                (
                    pd.to_numeric(frame[body.times[k]], errors="coerce"),
                    pd.to_numeric(frame[body.mapping[k]], errors="coerce"),
                )
                for k in NAMES
            ]
            flights.append(align(Path(file.name).stem, signals))
        groups = split(flights, assignments=body.assignments)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    id = uuid.uuid4().hex
    datasets[id] = (flights, groups)
    return {
        "dataset": id,
        "flights": len(flights),
        "split": {s: [f.name for f in fs] for s, fs in groups.items()},
    }


class RunRequest(BaseModel):
    source: Literal["bundled", "pulse", "flightLike", "user"] = "bundled"
    dataset: Optional[str] = None
    steps: int = Field(default=400, ge=20, le=2000)
    horizon: int = Field(default=192, ge=16, le=512)
    hidden: int = Field(default=16, ge=4, le=64)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    methods: list[Literal["oe", "neural", "stable", "null_rate"]] = Field(
        default=["oe", "neural"], min_length=1, max_length=3
    )
    syntheticTrainFlights: int = Field(default=0, ge=0, le=20)
    nullObserver: Literal["legacy", "stationary", "envelope"] = "legacy"
    nullAdaptive: bool = True
    nullTau: float = Field(default=2.0, gt=0, le=60)
    include_test: bool = False
    ddv_basis: Literal["rich", "legacy"] = "rich"
    warm_steps: int = Field(default=100, ge=20, le=1000)


@app.post("/api/run")
def start(body: RunRequest):
    if body.syntheticTrainFlights and body.source != "user":
        raise HTTPException(
            400,
            "Synthetic TRAIN augmentation requires user data; choose 0 for synthetic-only experiments",
        )
    if body.source == "user" and body.dataset not in datasets:
        raise HTTPException(400, "Import your CSV flights first")
    with lock:
        if any(j["status"] == "running" for j in jobs.values()):
            raise HTTPException(409, "A run is already active")
        id = uuid.uuid4().hex
        jobs[id] = {"status": "running", "log": [], "cancel": False}

    def worker():
        def log(message):
            jobs[id]["log"].append(str(message))
            if jobs[id]["cancel"]:
                raise RuntimeError("Cancelled at a safe stage boundary")

        try:
            flights, groups = (
                datasets[body.dataset]
                if body.source == "user"
                else (
                    bundled()
                    if body.source == "bundled"
                    else pulse_flights(
                        profile="flightLike"
                        if body.source == "flightLike"
                        else "pulse",
                        seed=621 if body.source == "flightLike" else 123,
                    )
                )
            )
            result, _, _ = run(
                flights,
                groups,
                source=body.source,
                steps=body.steps,
                horizon=body.horizon,
                hidden=body.hidden,
                seed=body.seed,
                methods=body.methods,
                include_test=body.include_test,
                log=log,
                ddv_basis=body.ddv_basis,
                warm_steps=body.warm_steps,
                syntheticTrainFlights=body.syntheticTrainFlights,
                syntheticProfile="flightLike"
                if body.source == "flightLike"
                else "pulse",
                nullAdaptive=body.nullAdaptive,
                nullObserver=body.nullObserver,
                nullTau=body.nullTau,
            )
            jobs[id].update(status="done", result=result)
        except Exception as e:
            jobs[id].update(
                status="cancelled" if jobs[id]["cancel"] else "failed", error=str(e)
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"job": id}


@app.get("/api/jobs/{id}")
def status(id: str):
    if id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[id]


@app.post("/api/jobs/{id}/cancel")
def cancel(id: str):
    if id not in jobs:
        raise HTTPException(404, "Unknown job")
    jobs[id]["cancel"] = True
    return {"message": "Cancellation requested; next safe stage boundary"}


def folder(id):
    if not id.replace("_", "").isdigit():
        raise HTTPException(400, "Invalid result identifier")
    p = RESULTS / id
    if not (p / "result.json").is_file():
        p = REFERENCES / id
    if not (p / "result.json").is_file():
        raise HTTPException(404, "Result not found")
    return p


@app.get("/api/results")
def results():
    rows = []
    paths = {p.parent.name: p for p in REFERENCES.glob("*/result.json")}
    paths.update({p.parent.name: p for p in RESULTS.glob("*/result.json")})
    for _, p in sorted(paths.items(), reverse=True):
        r = json.loads(p.read_text())
        rows.append({k: r[k] for k in ("id", "selected", "config", "notice")})
    return rows


@app.get("/api/results/{id}")
def result(id: str):
    return json.loads((folder(id) / "result.json").read_text())


@app.get("/api/results/{id}/files/{name}")
def artifact(id: str, name: str):
    if name not in ("models.json", "comparison.xlsx", "result.json"):
        raise HTTPException(404, "Unknown artifact")
    return FileResponse(folder(id) / name, filename=name)


@app.get("/api/results/{id}/flights")
def flight_names(id: str):
    return json.loads((folder(id) / "flights.json").read_text())


@app.get("/api/results/{id}/plot")
def plot(id: str, flight: int = 0, model: Optional[str] = None):
    p = folder(id)
    models = json.loads((p / "models.json").read_text())
    r = json.loads((p / "result.json").read_text())
    model = model or r["selected"]
    if model not in models:
        raise HTTPException(400, "Unknown model")
    if (p / "flights.npz").is_file():
        with np.load(p / "flights.npz", allow_pickle=False) as z:
            key = f"f{flight}"
            if key not in z:
                raise HTTPException(400, "Unknown flight")
            a = z[key].copy()
    else:
        flights = reference_flights(r)
        if flight < 0 or flight >= len(flights):
            raise HTTPException(400, "Unknown flight")
        f = flights[flight]
        a = np.column_stack([f.time, f.values])
    ddv, x, angle = rollout(models[model], a[:, 1], a[0, 3])
    # Bin extrema from measured and predicted channels preserve peaks in preview.
    ix = set([0, len(a) - 1])
    for part in np.array_split(np.arange(len(a)), min(400, len(a))):
        for y in (a[:, 2], ddv, a[:, 3], x, a[:, 4], angle):
            valid = np.where(np.isfinite(y[part]))[0]
            if len(valid):
                v = part[valid]
                ix.add(int(v[np.argmin(y[v])]))
                ix.add(int(v[np.argmax(y[v])]))
    ix = np.array(sorted(ix))
    return serial(
        {
            "time": a[ix, 0],
            "measured": a[ix, 2:5].T,
            "predicted": np.stack([ddv[ix], x[ix], angle[ix]]),
            "samples": len(a),
            "shown": len(ix),
        }
    )


_reference_cache = {}


def reference_flights(result):
    """Shipped public synthetic reports share raw data rather than duplicate it."""
    source = result["config"]["source"]
    if source not in ("bundled", "pulse", "flightLike"):
        raise HTTPException(404, "Local raw data missing")
    if source not in _reference_cache:
        if source == "bundled":
            flights, _ = bundled()
        elif source == "flightLike":
            flights, _ = pulse_flights(
                profile="flightLike", seed=result["config"]["seed"]
            )
        else:
            meta = json.loads((ROOT / "data/synthetic/pulse_hold_v1.json").read_text())
            with np.load(
                ROOT / "data/synthetic/pulse_hold_v1.npz", allow_pickle=False
            ) as raw:
                flights = [
                    Flight(name, raw[f"f{i}"][:, 0], raw[f"f{i}"][:, 1:])
                    for i, name in enumerate(meta["names"])
                ]
        _reference_cache[source] = flights
    flights = _reference_cache[source]
    if fingerprint(flights) != result["dataset_sha256"]:
        raise HTTPException(409, "Reference raw-data fingerprint mismatch")
    return flights
