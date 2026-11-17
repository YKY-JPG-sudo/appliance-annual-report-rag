#!/usr/bin/env python3
"""Run the fixed question set and persist retrieval/answer evidence."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.rag import HybridRetriever, ask


QUESTIONS = ROOT / "evaluation" / "questions.json"
OUT_JSON = ROOT / "outputs" / "evaluation_results.json"
OUT_MD = ROOT / "outputs" / "evaluation.md"


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    results = []

    for item in questions:
        started = time.perf_counter()
        answer = ask(retriever, item["question"])
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        keyword_hits = [
            keyword
            for keyword in item["expected_keywords"]
            if keyword in answer["answer"]
        ]
        correct = len(keyword_hits) == len(item["expected_keywords"])
        recalled_pages = {
            (chunk["company"], chunk["page"])
            for chunk in answer["chunks"]
        }
        source_pages = {
            (citation["company"], citation["page"])
            for citation in answer["citations"]
        }
        relevant_pages = {
            (location["company"], location["page"])
            for location in item["relevant_pages"]
        }
        results.append(
            {
                **item,
                "answer": answer["answer"],
                "mode": answer["mode"],
                "correct": correct,
                "keyword_hits": keyword_hits,
                "elapsed_ms": elapsed_ms,
                "relevant_pages_recalled": sorted(
                    [{"company": c, "page": p} for c, p in recalled_pages & relevant_pages],
                    key=lambda x: (x["company"], x["page"]),
                ),
                "relevant_pages_cited": sorted(
                    [{"company": c, "page": p} for c, p in source_pages & relevant_pages],
                    key=lambda x: (x["company"], x["page"]),
                ),
                "recalled_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                        "company": chunk["company"],
                        "page": chunk["page"],
                        "chapter": chunk["chapter"],
                        "score": chunk["score"],
                        "dense_score": chunk["dense_score"],
                        "bm25_score": chunk["bm25_score"],
                        "source_url": chunk["source_url"],
                    }
                    for rank, chunk in enumerate(answer["chunks"], start=1)
                ],
                "citations": answer["citations"],
            }
        )
        print(
            f"{item['id']} {'PASS' if correct else 'FAIL'} "
            f"{len(keyword_hits)}/{len(item['expected_keywords'])} {elapsed_ms} ms",
            flush=True,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(results),
        "correct": sum(result["correct"] for result in results),
        "results": results,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 年报问答评测",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 自动判分：{payload['correct']}/{payload['questions']}",
        "- 口径：期望答案中的关键事实全部出现在最终答案中才算通过；同时记录召回块与引用位置。",
        "",
        "| 题号 | 类型 | 问题 | 结果 | 命中 | 召回的正确页 | 引用正确页 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        recalled = ", ".join(
            f"{item['company']} p.{item['page']}"
            for item in result["relevant_pages_recalled"]
        ) or "无"
        cited = ", ".join(
            f"{item['company']} p.{item['page']}"
            for item in result["relevant_pages_cited"]
        ) or "无"
        lines.append(
            "| {id} | {type} | {question} | {status} | {hits}/{total} | {recalled} | "
            "{cited} | {elapsed} ms |".format(
                id=result["id"],
                type=result["type"],
                question=markdown_escape(result["question"]),
                status="通过" if result["correct"] else "未通过",
                hits=len(result["keyword_hits"]),
                total=len(result["expected_keywords"]),
                recalled=markdown_escape(recalled),
                cited=markdown_escape(cited),
                elapsed=result["elapsed_ms"],
            )
        )

    lines.extend(["", "## 逐题记录", ""])
    for result in results:
        lines.extend(
            [
                f"### {result['id']} {result['question']}",
                "",
                f"**答案**：{result['answer']}",
                "",
                "**召回 Top-8**：" + "；".join(
                    f"{chunk['company']} p.{chunk['page']} ({chunk['chunk_id']})"
                    for chunk in result["recalled_chunks"][:8]
                ),
                "",
                f"**判定**：{'答对' if result['correct'] else '未全对'}。"
                f" 命中关键词 {result['keyword_hits']}。",
                "",
                f"**复盘**：{result['review_note'] or '答案事实与引用页码一致，无已知错误。'}",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT_JSON.relative_to(ROOT)} 和 {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
