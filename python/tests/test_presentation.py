import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from pptx import Presentation
from pptx.util import Inches
import build_presentation as module


class PresentationTests(unittest.TestCase):
    def test_native_content_and_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            template = tmp / "template.pptx"
            p = Presentation()
            p.slide_width = Inches(12)
            p.slide_height = Inches(7)
            p.slides.add_slide(p.slide_layouts[0]).shapes.title.text = "DELETE ME"
            p.save(template)
            out = module.build(str(template), str(tmp / "out.pptx"))
            p = Presentation(out)
            self.assertEqual(len(p.slides), len(module.CONTENT))
            self.assertEqual(p.slide_width, Inches(12))
            charts = 0
            for i, s in enumerate(p.slides, 1):
                self.assertIsNotNone(s.shapes.title)
                self.assertNotEqual(s.shapes.title.text, "DELETE ME")
                body = [sh for sh in s.shapes if sh.name == "content_bullets"]
                self.assertLessEqual(len(body), 1)
                for sh in body:
                    self.assertIn("buChar", sh._element.xml)
                self.assertIn("slidenum", s._element.xml)
                for sh in s.shapes:
                    self.assertGreaterEqual(sh.left, 0)
                    self.assertGreaterEqual(sh.top, 0)
                    self.assertLessEqual(sh.left + sh.width, p.slide_width + 5)
                    self.assertLessEqual(sh.top + sh.height, p.slide_height + 5)
                    charts += int(sh.has_chart)
            self.assertEqual(
                charts,
                sum(
                    item["visual"] in ("hold", "step", "vibration", "offset")
                    for item in module.CONTENT
                ),
            )
            with ZipFile(out) as z:
                self.assertIn(b"Franklin Gothic Medium", z.read("ppt/theme/theme1.xml"))
            font = module.SETTINGS["font"]
            try:
                module.SETTINGS["font"] = None
                out = module.build(str(template), str(tmp / "inherit.pptx"))
                with ZipFile(template) as a, ZipFile(out) as b:
                    self.assertEqual(
                        a.read("ppt/theme/theme1.xml"), b.read("ppt/theme/theme1.xml")
                    )
            finally:
                module.SETTINGS["font"] = font
