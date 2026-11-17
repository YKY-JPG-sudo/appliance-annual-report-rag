#!/usr/bin/env python3
"""Extract report pages/tables, create provenance-preserving chunks, and build indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import pdfplumber
from fastembed import TextEmbedding
from pypdf import PdfReader
from sklearn.preprocessing import normalize

from src.config import CHUNKS_PATH, DATA_DIR, INDEX_DIR, MANIFEST_PATH, PAGES_PATH, TABLES_PATH, ROOT
from src.text import infer_chapter, normalize_text, split_sentences, tokenize


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
MODEL_CACHE = ROOT / ".cache" / "fastembed"


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = []
    for row in rows:
        values = [
            re.sub(r"\s+", " ", str(cell or "")).strip()
            for cell in row
        ]
        if any(values):
            cleaned.append(values)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    if not any(header):
        header = [f"列{i + 1}" for i in range(width)]
        body = cleaned
    else:
        body = cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def window_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []

    sentences = split_sentences(text)
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current}\n{sentence}".strip()
            continue
        if current:
            windows.append(current)
            tail = current[-overlap:]
            current = f"{tail}\n{sentence}".strip()
        else:
            windows.extend(text[i : i + size] for i in range(0, len(text), size - overlap))
            return windows
        while len(current) > size:
            windows.append(current[:size])
            current = current[size - overlap :]
    if current:
        windows.append(current)
    return windows


def extract_report(item: dict) -> tuple[list[dict], list[dict], list[dict]]:
    path = ROOT / item["local_path"]
    reader = PdfReader(path)
    pages: list[dict] = []
    tables: list[dict] = []
    chunks: list[dict] = []
    current_chapter = "报告首页"

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = normalize_text(page.extract_text(x_tolerance=2, y_tolerance=3) or "")
            for line in raw_text.splitlines()[:30]:
                current_chapter = infer_chapter(line, current_chapter)

            page_tables = []
            try:
                raw_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    }
                )
            except Exception:
                raw_tables = []

            for table_index, rows in enumerate(raw_tables, start=1):
                markdown = table_to_markdown(rows)
                if len(markdown) < 20:
                    continue
                table = {
                    "table_id": stable_id(item["code"], page_number, table_index, markdown),
                    "company": item["name"],
                    "code": item["code"],
                    "report_title": item["report_title"],
                    "page": page_number,
                    "chapter": current_chapter,
                    "table_index": page_number * 100 + table_index,
                    "markdown": markdown,
                    "rows": rows,
                    "source_url": item["detail_url"],
                    "pdf_url": item["pdf_url"],
                }
                tables.append(table)
                page_tables.append(markdown)

            page_record = {
                "company": item["name"],
                "code": item["code"],
                "report_title": item["report_title"],
                "page": page_number,
                "chapter": current_chapter,
                "text": raw_text,
                "table_count": len(page_tables),
            }
            pages.append(page_record)

            content = raw_text
            if page_tables:
                content += "\n\n本页表格：\n" + "\n\n".join(page_tables)
            for window_index, window in enumerate(window_text(content), start=1):
                chunk_id = stable_id(item["code"], page_number, window_index, window[:80])
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "company": item["name"],
                        "code": item["code"],
                        "segment": item["segment"],
                        "report_title": item["report_title"],
                        "report_period": item["report_period"],
                        "page": page_number,
                        "chapter": current_chapter,
                        "text": window,
                        "has_table": bool(page_tables and (
                            "本页表格：" in window or window_index == len(window_text(content))
                        )),
                        "source_url": item["detail_url"],
                        "pdf_url": item["pdf_url"],
                    }
                )
        item["pdf_pages"] = len(pdf.pages)
        item["pypdf_pages"] = len(reader.pages)
    return pages, tables, chunks


def build_dense(chunks: list[dict]) -> None:
    texts = [
        f"{chunk['company']}｜{chunk['chapter']}｜{chunk['text']}"
        for chunk in chunks
    ]
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(MODEL_CACHE))
    embeddings = np.asarray(list(model.embed(texts, batch_size=32)), dtype=np.float32)
    embeddings = normalize(embeddings)
    np.save(INDEX_DIR / "dense_embeddings.npy", embeddings)
    (INDEX_DIR / "embedding_model.txt").write_text(EMBEDDING_MODEL + "\n", encoding="utf-8")


def build_sparse(chunks: list[dict]) -> None:
    documents = [
        " ".join(tokenize(f"{c['company']} {c['chapter']} {c['text']}"))
        for c in chunks
    ]
    (INDEX_DIR / "bm25_tokens.json").write_text(
        json.dumps(documents, ensure_ascii=False),
        encoding="utf-8",
    )
    counts = Counter(token for doc in documents for token in doc.split())
    stats = {
        "documents": len(documents),
        "vocabulary": len(counts),
        "avg_doc_tokens": round(sum(len(doc.split()) for doc in documents) / max(len(documents), 1), 2),
    }
    (INDEX_DIR / "bm25_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--reuse-extraction",
        action="store_true",
        help="Reuse data/*.jsonl and only rebuild the indexes.",
    )
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.reuse_extraction:
        all_pages = [
            json.loads(line)
            for line in PAGES_PATH.read_text(encoding="utf-8").splitlines()
        ]
        all_tables = [
            json.loads(line)
            for line in TABLES_PATH.read_text(encoding="utf-8").splitlines()
        ]
        all_chunks = [
            json.loads(line)
            for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        ]
    else:
        all_pages = []
        all_tables = []
        all_chunks = []
        for index, item in enumerate(manifest, start=1):
            print(f"[{index:02d}/{len(manifest)}] 提取 {item['name']}...", flush=True)
            pages, tables, chunks = extract_report(item)
            all_pages.extend(pages)
            all_tables.extend(tables)
            all_chunks.extend(chunks)
            print(f"    {len(pages)} 页, {len(tables)} 张表, {len(chunks)} 块")

        for path, records in (
            (PAGES_PATH, all_pages),
            (TABLES_PATH, all_tables),
            (CHUNKS_PATH, all_chunks),
        ):
            with path.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("生成中文 BGE 向量索引（首次运行会下载约 90 MB 模型）...")
    build_dense(all_chunks)
    print("生成 BM25 词项...")
    build_sparse(all_chunks)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "companies": len(manifest),
        "pages": len(all_pages),
        "tables": len(all_tables),
        "chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    (DATA_DIR / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

