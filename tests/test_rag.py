from __future__ import annotations

import json
import unittest

from src.config import CHUNKS_PATH, DATA_DIR, MANIFEST_PATH, TABLES_PATH
from src.rag import HybridRetriever, ask


class CorpusTests(unittest.TestCase):
    def test_manifest_has_twelve_companies_and_sources(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 12)
        for item in manifest:
            self.assertTrue(item["detail_url"].startswith("https://www.cninfo.com.cn/"))
            self.assertTrue(item["pdf_url"].startswith("https://static.cninfo.com.cn/"))
            self.assertGreaterEqual(len(item["sha256"]), 64)

    def test_chunks_keep_page_and_section_provenance(self) -> None:
        lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(lines), 6000)
        for line in lines[:50]:
            chunk = json.loads(line)
            self.assertTrue(chunk["company"])
            self.assertGreater(chunk["page"], 0)
            self.assertTrue(chunk["chapter"])
            self.assertTrue(chunk["source_url"])

    def test_table_cells_are_row_column_preserved(self) -> None:
        table = json.loads(TABLES_PATH.read_text(encoding="utf-8").splitlines()[0])
        self.assertIsInstance(table["rows"], list)
        self.assertTrue(table["rows"])
        self.assertTrue(any(any(cell for cell in row) for row in table["rows"]))
        self.assertIn("|", table["markdown"])

    def test_metrics_cover_every_company(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        metrics = json.loads((DATA_DIR / "financial_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {item["name"] for item in manifest},
            {item["company"] for item in metrics},
        )


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = HybridRetriever()

    def test_table_row_question(self) -> None:
        answer = ask(self.retriever, "九阳股份2025年境外销售收入及同比变化是多少？")
        self.assertIn("825,127,010.61", answer["answer"])
        self.assertIn("-48.83%", answer["answer"])
        self.assertIn(("九阳股份", 13), {
            (item["company"], item["page"]) for item in answer["citations"]
        })

    def test_cross_company_ranking(self) -> None:
        answer = ask(
            self.retriever,
            "12家公司中，2025年营业收入和归母净利润最高的分别是谁？",
        )
        self.assertIn("美的集团（4,564.52", answer["answer"])
        self.assertIn("海尔智家（3,023.47", answer["answer"])
        self.assertEqual(answer["mode"], "structured_metric")

    def test_cross_company_threshold(self) -> None:
        answer = ask(
            self.retriever,
            "哪些公司经营现金流同比增长率超过50%？",
        )
        for company in ("九阳股份", "飞科电器", "小熊电器", "格力电器"):
            self.assertIn(company, answer["answer"])


if __name__ == "__main__":
    unittest.main()

