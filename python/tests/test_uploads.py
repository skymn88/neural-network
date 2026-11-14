import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from actuator.data import NAMES
from actuator.uploads import SignalReducer, read_selected, receive_upload


class UploadTests(unittest.TestCase):
    def test_chunk_phase_and_antialias(self):
        t = np.arange(24000) / 1000
        x = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 200 * t)
        a = SignalReducer()
        a.push(t, x)
        ta, xa = a.finish()
        b = SignalReducer()
        for start in range(0, len(t), 731):
            b.push(t[start : start + 731], x[start : start + 731])
        tb, xb = b.finish()
        np.testing.assert_allclose(ta, tb, atol=1e-9)
        np.testing.assert_allclose(xa, xb, atol=1e-7)
        self.assertLess(np.std(xa[64:-64] - np.sin(2 * np.pi * 2 * ta[64:-64])), 0.005)

    def test_usecols_and_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            t = np.arange(4096) / 128
            pd.DataFrame(dict(t=t, signal=np.sin(t), unused=["bad"] * len(t))).to_csv(
                path, index=False
            )
            f = read_selected(
                path,
                "f",
                {k: "signal" for k in NAMES},
                {k: "t" for k in NAMES},
                chunksize=513,
            )
            self.assertGreater(len(f.time), 2000)
            self.assertAlmostEqual(np.median(np.diff(f.time)), 1 / 64)

    def test_stream_size_and_partial_cleanup(self):
        class Request:
            headers = {"content-type": "multipart/form-data; boundary=X"}

            async def stream(self):
                data = b'--X\r\nContent-Disposition: form-data; name="file"; filename="f.csv"\r\n\r\na,b\n1,2\n\r\n--X--\r\n'
                for i in range(0, len(data), 7):
                    yield data[i : i + 7]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("actuator.uploads.UPLOADS", Path(tmp)),
        ):
            item = asyncio.run(receive_upload(Request()))
            self.assertEqual(Path(item["path"]).read_bytes(), b"a,b\n1,2\n")
            Path(item["path"]).unlink()
            with patch.dict(os.environ, {"ACTUATOR_MAX_UPLOAD_MB": "0.000001"}):
                with self.assertRaises(OverflowError):
                    asyncio.run(receive_upload(Request()))
            self.assertFalse(list(Path(tmp).iterdir()))
