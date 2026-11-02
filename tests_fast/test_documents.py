import tempfile
from pathlib import Path
import unittest

try:
    import docx
except ImportError:
    docx = None


@unittest.skipIf(docx is None, 'python-docx is needed only for document generation')
class DocumentTests(unittest.TestCase):
    def setUp(self):
        from scripts.fast_build_documents import ROOT, TEMPLATE, build
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'flow.docx'
        self.source = ROOT / 'docs/FAST_REVIEW_FLOW.md'
        self.template = docx.Document(TEMPLATE)
        build(self.source, self.path)
        self.document = docx.Document(self.path)

    def test_original_page_geometry_and_title(self):
        original, actual = self.template.sections[0], self.document.sections[0]
        for key in ('page_width', 'page_height', 'top_margin', 'bottom_margin',
                    'left_margin', 'right_margin', 'header_distance', 'footer_distance'):
            self.assertEqual(getattr(original, key), getattr(actual, key))
        title = self.document.paragraphs[0]
        self.assertEqual(title.text, '7. 测试流程')
        self.assertEqual(title.runs[0].font.size.pt, 14)
        self.assertEqual(str(title.runs[0].font.color.rgb), '000000')
        self.assertNotEqual(str(title.alignment), 'CENTER (1)')

    def test_commands_are_exact_and_white_on_black(self):
        from docx.oxml.ns import qn
        from scripts.fast_build_documents import blocks
        expected = [text for kind, text in blocks(self.source.read_text(encoding='utf-8'))
                    if kind == 'code']
        commands = [p for p in self.document.paragraphs if p.style.name == 'Command']
        self.assertEqual(expected, [p.text for p in commands])
        for paragraph in commands:
            self.assertEqual(paragraph._p.pPr.find(qn('w:shd')).get(qn('w:fill')), '000000')
            self.assertTrue(paragraph.paragraph_format.keep_together)
            for run in paragraph.runs:
                self.assertEqual(str(run.font.color.rgb), 'FFFFFF')
                self.assertEqual(run.font.name, '宋体')
                self.assertEqual(run.font.size.pt, 12)

    def test_body_and_step_styles(self):
        headings = [p for p in self.document.paragraphs if p.style.name == 'Heading 1']
        self.assertEqual(len(headings), 5)
        self.assertEqual([p.text[0] for p in headings], list('12345'))
        for paragraph in self.document.paragraphs[2:]:
            if paragraph.style.name == 'Command':
                continue
            for run in paragraph.runs:
                self.assertEqual(str(run.font.color.rgb), '000000')
                self.assertIn(run.font.name, ('宋体', 'Consolas'))
        text = '\n'.join(p.text for p in self.document.paragraphs)
        self.assertNotIn('ml_fitted_001', text)
        self.assertIn('全量历史证据复核', text)
        self.assertIn('runs/fast_001/report.html', text)
        for name in ('signal_report', 'calibration_report', 'warning_report'):
            self.assertIn(name + '.html', text)
            self.assertIn(name + '.md', text)


if __name__ == '__main__':
    unittest.main()
