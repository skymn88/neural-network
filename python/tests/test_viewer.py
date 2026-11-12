import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "viewer_export", Path(__file__).parents[1] / "viewer/export_results.py"
)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTests(unittest.TestCase):
    def test_static_export_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "refs" / "1"
            ref.mkdir(parents=True)
            result = {
                "id": "1",
                "config": {"source": "pulse"},
                "selected": "NN_joint",
                "validation_scores": {"NN_joint": 1},
            }
            (ref / "result.json").write_text(json.dumps(result))
            (ref / "flights.json").write_text('["flight1"]')
            (ref / "comparison.xlsx").write_bytes(b"fixture")
            with (
                patch.object(viewer, "REFERENCES", ref.parent),
                patch.object(
                    viewer,
                    "plot",
                    return_value={"time": [0.0, 1.0], "predicted": [[0.123456789]]},
                ),
            ):
                viewer.export(root / "out")
            self.assertEqual(
                json.loads((root / "out/catalog.json").read_text())[0]["id"], "1"
            )
            self.assertEqual(
                json.loads((root / "out/1/flight_0_NN_joint.json").read_text())[
                    "predicted"
                ][0][0],
                0.1234568,
            )
            self.assertTrue((root / "out/index.html").is_file())

    def test_user_data_never_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "refs" / "1"
            ref.mkdir(parents=True)
            (ref / "result.json").write_text('{"config":{"source":"user"}}')
            with (
                patch.object(viewer, "REFERENCES", ref.parent),
                self.assertRaisesRegex(ValueError, "synthetic"),
            ):
                viewer.export(root / "out")
