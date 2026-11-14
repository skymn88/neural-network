"""Bounded multipart-to-disk upload and selected-column streaming resampling."""

import os, uuid, time
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from .data import ROOT, FS, NAMES, align

UPLOADS = ROOT / "python/uploads"


def upload_limit():
    value = float(os.environ.get("ACTUATOR_MAX_UPLOAD_MB", "512"))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("ACTUATOR_MAX_UPLOAD_MB must be positive")
    return int(value * 1024**2)


async def receive_upload(request):
    from python_multipart import MultipartParser
    from python_multipart.multipart import parse_options_header

    limit = upload_limit()
    content_type, options = parse_options_header(
        request.headers.get("content-type", "")
    )
    if content_type != b"multipart/form-data" or b"boundary" not in options:
        raise ValueError("multipart/form-data with file field required")
    if int(request.headers.get("content-length", "0")) > limit + 65536:
        raise OverflowError("Upload exceeds configured MiB limit")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    for stale in UPLOADS.glob("*.csv"):
        if time.time() - stale.stat().st_mtime > 86400:
            stale.unlink(missing_ok=True)
    id = uuid.uuid4().hex
    path = UPLOADS / (id + ".csv")
    state = dict(
        headers={},
        field=b"",
        value=b"",
        size=0,
        parts=0,
        ended=False,
        file=None,
        name=None,
        header_bytes=0,
    )

    def begin():
        state["parts"] += 1
        if state["parts"] > 1:
            raise ValueError("Upload one CSV per request")

    def header_field(data, start, end):
        state["header_bytes"] += end - start
        if state["header_bytes"] > 16384:
            raise ValueError("Multipart headers too large")
        state["field"] += data[start:end]

    def header_value(data, start, end):
        state["header_bytes"] += end - start
        if state["header_bytes"] > 16384:
            raise ValueError("Multipart headers too large")
        state["value"] += data[start:end]

    def header_end():
        state["headers"][state["field"].lower()] = state["value"]
        state["field"] = state["value"] = b""

    def headers_done():
        _, opts = parse_options_header(
            state["headers"].get(b"content-disposition", b"")
        )
        if opts.get(b"name") != b"file" or b"filename" not in opts:
            raise ValueError("Expected CSV file field")
        name = (
            opts[b"filename"]
            .decode("utf-8", errors="replace")
            .replace("\\", "/")
            .split("/")[-1]
        )
        if not name.lower().endswith(".csv"):
            raise ValueError("CSV extension required")
        state["name"] = name
        state["file"] = path.open("wb")

    def data(data, start, end):
        state["size"] += end - start
        if state["size"] > limit:
            raise OverflowError("Upload exceeds configured MiB limit")
        if state["file"] is None:
            raise ValueError("File headers missing")
        state["file"].write(data[start:end])

    def ended():
        state["ended"] = True

    parser = MultipartParser(
        options[b"boundary"],
        dict(
            on_part_begin=begin,
            on_header_field=header_field,
            on_header_value=header_value,
            on_header_end=header_end,
            on_headers_finished=headers_done,
            on_part_data=data,
            on_end=ended,
        ),
    )
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit + 65536:
                raise OverflowError("Upload exceeds configured MiB limit")
            parser.write(chunk)
        parser.finalize()
        if not state["ended"] or not state["size"]:
            raise ValueError("Incomplete or empty multipart upload")
    except BaseException:
        if state["file"]:
            state["file"].close()
        path.unlink(missing_ok=True)
        raise
    state["file"].close()
    return dict(id=id, name=state["name"], bytes=state["size"], path=str(path))


class SignalReducer:
    """Polyphase anti-aliasing with fixed global phase and bounded overlap."""

    def __init__(self):
        self.t = np.empty(0)
        self.x = np.empty(0)
        self.origin = None
        self.dt = None
        self.next = 0
        self.out_t = []
        self.out_x = []
        self.last = None

    def push(self, t, x, final=False):
        t = np.asarray(t, float)
        x = np.asarray(x, float)
        ok = np.isfinite(t) & np.isfinite(x)
        t = t[ok]
        x = x[ok]
        if len(t):
            if np.any(np.diff(t) < 0) or (self.last is not None and t[0] < self.last):
                raise ValueError(
                    "Selected timestamps must be monotonic; sort the CSV first"
                )
            keep = np.r_[True, np.diff(t) > 0]
            t = t[keep]
            x = x[keep]
            if self.last is not None:
                keep = t > self.last
                t = t[keep]
                x = x[keep]
            if len(t):
                self.last = t[-1]
                self.t = np.r_[self.t, t]
                self.x = np.r_[self.x, x]
        if self.dt is None:
            if len(self.t) < 10:
                if final:
                    raise ValueError("Too few finite samples in a selected channel")
                return
            self.origin = self.t[0]
            self.dt = float(np.median(np.diff(self.t)))
        if len(self.t) < 2:
            return
        if np.max(np.diff(self.t)) > 5 * self.dt:
            raise ValueError("Selected signal contains a long time gap")
        end = self.t[-1] if final else self.t[-1] - 0.5
        stop = int(np.floor((end - self.origin) * FS + 1e-8)) + 1
        if stop <= self.next:
            return
        if 1 / self.dt > FS * 1.05:
            ratio = Fraction(FS * self.dt).limit_denominator(10000)
            start = int(np.ceil((self.t[0] - self.origin) / self.dt - 1e-7))
            start = int(np.ceil(start / ratio.denominator)) * ratio.denominator
            last = int(np.floor((self.t[-1] - self.origin) / self.dt + 1e-7))
            grid = self.origin + np.arange(start, last + 1) * self.dt
            if len(grid) < 2:
                return
            values = resample_poly(
                np.interp(grid, self.t, self.x),
                ratio.numerator,
                ratio.denominator,
                padtype="constant",
            )
            times = (
                grid[0]
                + np.arange(len(values)) * self.dt * ratio.denominator / ratio.numerator
            )
        else:
            times, values = self.t, self.x
        target = self.origin + np.arange(self.next, stop) / FS
        if target[0] < times[0] - 1e-8:
            raise ValueError("Resampling overlap is insufficient")
        self.out_t.append(target)
        self.out_x.append(np.interp(target, times, values))
        self.next = stop
        cutoff = self.origin + self.next / FS - 1.0
        keep = max(0, np.searchsorted(self.t, cutoff) - 2)
        self.t = self.t[keep:]
        self.x = self.x[keep:]

    def finish(self):
        self.push([], [], True)
        return np.concatenate(self.out_t), np.concatenate(self.out_x)


def read_selected(path, name, mapping, times, separator=",", chunksize=65536):
    usecols = sorted(set(mapping.values()) | set(times.values()))
    if set(mapping) != set(NAMES) or set(times) != set(NAMES):
        raise ValueError("Map all four signals and their time columns")
    reducers = {k: SignalReducer() for k in NAMES}
    for frame in pd.read_csv(path, sep=separator, usecols=usecols, chunksize=chunksize):
        columns = {
            c: pd.to_numeric(frame[c], errors="coerce").to_numpy() for c in usecols
        }
        for k in NAMES:
            reducers[k].push(columns[times[k]], columns[mapping[k]])
    return align(name, [reducers[k].finish() for k in NAMES])
