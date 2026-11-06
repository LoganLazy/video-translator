import tempfile
import unittest
from pathlib import Path

from study_pack import make_chapters, write_study_pack
from study_ai import enrich_chapters


class StudyPackTests(unittest.TestCase):
    def setUp(self):
        self.segs = [(0, 3, "first"), (4, 8, "second"), (500, 505, "third")]
        self.pairs = [
            ("first", "第一部分内容。"),
            ("second", "继续解释知识。"),
            ("third", "下一章节内容。"),
        ]

    def test_chapters_split_on_long_gap(self):
        chapters = make_chapters(self.segs, self.pairs, target_seconds=100)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1]["start"], 500)

    def test_page_contains_features_and_escapes_script_end(self):
        pairs = list(self.pairs)
        pairs[0] = ("first", "字幕 </script><script>alert(1)</script>")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study.html"
            write_study_pack(Path("lesson.mp4"), self.segs, pairs, output)
            page = output.read_text(encoding="utf-8")
        self.assertIn("导出 Markdown", page)
        self.assertIn("继续上次进度", page)
        self.assertNotIn("</script><script>alert(1)</script>", page)

    def test_ai_outline_enriches_chapter(self):
        chapters = make_chapters(self.segs, self.pairs, target_seconds=100)

        def fake_chat(_messages):
            return '''```json
            {"chapters":[{"number":1,"title":"核心概念", "summary":"解释基础知识。",
            "keywords":["概念","基础"],"questions":["核心概念是什么？"]}]}
            ```'''

        result = enrich_chapters(chapters, self.segs, self.pairs, fake_chat)
        self.assertEqual(result[0]["title"], "核心概念")
        self.assertEqual(result[0]["keywords"], ["概念", "基础"])
        self.assertEqual(result[1]["title"], chapters[1]["title"])


if __name__ == "__main__":
    unittest.main()
