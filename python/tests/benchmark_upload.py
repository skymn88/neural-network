"""Manual 101 MB multipart/import benchmark; requires local app on port 8771."""

import http.client
import json
import tempfile
import time
import argparse
from pathlib import Path
import numpy as np


def benchmark(port, pid):
    def memory():
        fields = dict(
            line.split(":", 1)
            for line in Path(f"/proc/{pid}/status").read_text().splitlines()
            if ":" in line
        )
        return {key: int(fields[key].split()[0]) * 1024 for key in ("VmRSS", "VmHWM")}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "101mb.csv"
        with path.open("w") as f:
            f.write("time,CoilCurrent,DDV,RAM,Deflection,unused\n")
            offset = 0
            while f.tell() < 101 * 1024**2:
                t = np.arange(offset, offset + 10000) / 1000
                rows = [
                    f"{a:.3f},{b:.8f},{0.8 * b:.8f},{10 * b:.8f},{-3 * b:.8f},"
                    + ("x" * 160)
                    + "\n"
                    for a, b in zip(t, np.sin(t))
                ]
                f.writelines(rows)
                offset += 10000
        # Exactly 101 MiB, keeping a complete final row and monotonic timestamps.
        target = 101 * 1024**2
        with path.open("r+b") as f:
            f.seek(target-1024)
            tail_block=f.read(1024)
            boundary=tail_block.rfind(b"\n",0,768)
            last_time=float(tail_block[:boundary].split(b"\n")[-1].split(b",")[0])
            cut=target-1024+boundary+1
            prefix=f"{last_time+.001:.3f},0,0,0,0,".encode()
            f.seek(cut);f.truncate()
            f.write(prefix+b"x"*(target-cut-len(prefix)-1)+b"\n")
            offset=round(last_time*1000)+2
        before = memory()
        start = time.perf_counter()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        head = b'--BENCH\r\nContent-Disposition: form-data; name="file"; filename="101mb.csv"\r\nContent-Type: text/csv\r\n\r\n'
        tail = b"\r\n--BENCH--\r\n"
        conn.putrequest("POST", "/api/upload")
        conn.putheader("Content-Type", "multipart/form-data; boundary=BENCH")
        conn.putheader(
            "Content-Length", str(len(head) + path.stat().st_size + len(tail))
        )
        conn.endheaders()
        conn.send(head)
        with path.open("rb") as f:
            while True:
                chunk = f.read(262144)
                if not chunk:
                    break
                conn.send(chunk)
        conn.send(tail)
        response = conn.getresponse()
        item = json.loads(response.read())
        assert response.status == 200, item
        uploaded = time.perf_counter()
        after_upload = memory()
        mapping = {k: k for k in ("CoilCurrent", "DDV", "RAM", "Deflection")}
        body = json.dumps(
            dict(ids=[item["id"]], mapping=mapping, times={k: "time" for k in mapping})
        )
        conn.request(
            "POST", "/api/import-upload", body, {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        result = json.loads(response.read())
        assert response.status == 200, result
        end = time.perf_counter()
        after = memory()
        record = dict(
            bytes=path.stat().st_size,
            input_rows=offset,
            upload_seconds=uploaded - start,
            import_seconds=end - uploaded,
            total_seconds=end - start,
            before=before,
            after_upload=after_upload,
            after=after,
            result=result,
            measurement="Linux process VmHWM includes API dependencies; chunked multipart over HTTP, usecols resampling to64Hz",
        )
        print(json.dumps(record, indent=2))
        return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--pid", type=int, required=True)
    a = p.parse_args()
    benchmark(a.port, a.pid)
