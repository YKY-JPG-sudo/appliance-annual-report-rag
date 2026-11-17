from __future__ import annotations

import json
import os
import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH, DATA_DIR, INDEX_DIR, ROOT
from src.text import normalize_text, split_sentences, tokenize


os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastembed import TextEmbedding


MODEL_CACHE = ROOT / ".cache" / "fastembed"
METRICS_PATH = DATA_DIR / "financial_metrics.json"
CONFIG_COMPANIES = json.loads(
    (ROOT / "config" / "companies.json").read_text(encoding="utf-8")
)
KNOWN_COMPANY_NAMES = {
    item["name"] for item in CONFIG_COMPANIES
}

METRIC_LABELS = {
    "revenue_yi": "营业收入",
    "net_profit_yi": "归母净利润",
    "operating_cash_flow_yi": "经营活动现金流净额",
    "revenue_growth_pct": "营业收入同比",
    "net_profit_growth_pct": "归母净利润同比",
    "operating_cash_flow_growth_pct": "经营现金流同比",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks = load_jsonl(CHUNKS_PATH)
        self.embeddings = np.load(INDEX_DIR / "dense_embeddings.npy")
        self.bm25_tokens = json.loads(
            (INDEX_DIR / "bm25_tokens.json").read_text(encoding="utf-8")
        )
        self.tokens = [document.split() for document in self.bm25_tokens]
        self.bm25 = BM25Okapi(self.tokens)
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.embedder = TextEmbedding(
            model_name=(INDEX_DIR / "embedding_model.txt").read_text(encoding="utf-8").strip(),
            cache_dir=str(MODEL_CACHE),
        )
        self.companies = sorted({chunk["company"] for chunk in self.chunks})

    def _global_scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        query_vector = np.asarray(list(self.embedder.embed([query]))[0], dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-9)
        dense = self.embeddings @ query_vector
        sparse = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        return dense, sparse

    def _search_indices(
        self,
        query: str,
        indices: np.ndarray,
        limit: int,
        company_names: set[str] | None = None,
    ) -> list[dict]:
        dense_all, sparse_all = self._global_scores(query)
        dense = dense_all[indices]
        sparse = sparse_all[indices]

        dense_order = np.argsort(-dense, kind="stable")
        sparse_order = np.argsort(-sparse, kind="stable")
        rrf = np.zeros(len(indices), dtype=np.float32)
        for rank, position in enumerate(dense_order[: min(200, len(indices))], start=1):
            rrf[position] += 1.0 / (60 + rank)
        for rank, position in enumerate(sparse_order[: min(200, len(indices))], start=1):
            rrf[position] += 1.0 / (60 + rank)

        query_lower = query.lower()
        for position, global_index in enumerate(indices):
            chunk = self.chunks[int(global_index)]
            if chunk["company"].lower() in query_lower:
                rrf[position] += 0.015
            title = chunk["report_title"].lower()
            if title and title in query_lower:
                rrf[position] += 0.01

        order = np.argsort(-rrf, kind="stable")[:limit]
        results = []
        for local_position in order:
            local_position = int(local_position)
            global_index = int(indices[local_position])
            item = dict(self.chunks[global_index])
            item["score"] = round(float(rrf[local_position]), 6)
            item["dense_score"] = round(float(dense[local_position]), 6)
            item["bm25_score"] = round(float(sparse[local_position]), 6)
            results.append(item)
        return results

    def search(
        self,
        query: str,
        companies: list[str] | None = None,
        top_k: int = 8,
        per_company: bool = False,
    ) -> list[dict]:
        selected = set(companies or [])
        indices = np.asarray(
            [
                index
                for index, chunk in enumerate(self.chunks)
                if not selected or chunk["company"] in selected
            ],
            dtype=np.int64,
        )
        if not len(indices):
            return []
        if not per_company or not selected:
            return self._search_indices(query, indices, top_k)

        results: list[dict] = []
        per_group = max(1, min(3, top_k // max(len(selected), 1)))
        for company in sorted(selected):
            company_indices = np.asarray(
                [index for index in indices if self.chunks[int(index)]["company"] == company],
                dtype=np.int64,
            )
            results.extend(
                self._search_indices(query, company_indices, per_group, selected)
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


def detect_companies(query: str, companies: list[dict]) -> list[dict]:
    found = [
        company
        for company in companies
        if company["name"] in query or company["code"] in query
    ]
    if found:
        return found
    if any(
        word in query
        for word in (
            "12家",
            "十二家",
            "这些公司",
            "各公司",
            "整个样本",
            "全样本",
            "哪些公司",
            "哪几家公司",
            "哪家公司",
        )
    ):
        return companies
    return []


def detect_metrics(query: str) -> list[str]:
    metrics = []
    if "业绩" in query and any(
        word in query for word in ("增长", "同比", "变化", "原因", "驱动")
    ):
        metrics.extend(["revenue_growth_pct", "net_profit_growth_pct"])
    if any(word in query for word in ("营业收入", "营收", "收入规模")):
        metrics.append("revenue_yi")
    if any(word in query for word in ("净利润", "归母净利", "盈利")):
        metric = "net_profit_growth_pct" if any(
            word in query for word in ("同比", "增速", "增长", "下降")
        ) else "net_profit_yi"
        metrics.append(metric)
    if any(word in query for word in ("经营现金流", "经营活动现金流", "现金流")):
        metric = "operating_cash_flow_growth_pct" if any(
            word in query for word in ("同比", "增速", "增长", "下降")
        ) else "operating_cash_flow_yi"
        metrics.append(metric)
    if "营业收入同比" in query or (
        "营业收入" in query and any(word in query for word in ("同比", "增速", "增长"))
    ):
        metrics.append("revenue_growth_pct")
    return list(dict.fromkeys(metrics))


def _format_metric(metric: str, value: float) -> str:
    if metric.endswith("_pct"):
        return f"{value:+.2f}%"
    return f"{value:,.2f} 亿元"


def _metric_citation(company: dict, company_map: dict[str, dict]) -> dict:
    source = company_map[company["company"]]
    return {
        "company": company["company"],
        "code": company["code"],
        "report_title": source["report_title"],
        "page": company["page"],
        "chapter": "第二节 公司简介和主要财务指标",
        "source_url": source["detail_url"],
        "pdf_url": source["pdf_url"],
        "kind": "structured_fact",
    }


def answer_metrics(
    query: str,
    metrics: list[str],
    matched_companies: list[dict],
    metrics_rows: list[dict],
    manifest: list[dict],
) -> dict:
    by_name = {row["company"]: row for row in metrics_rows}
    company_map = {item["name"]: item for item in manifest}
    rows = [by_name[company["name"]] for company in matched_companies]
    labels = [METRIC_LABELS[metric] for metric in metrics]

    wants_rank = any(
        word in query
        for word in ("最高", "最大", "最低", "最小", "排名", "前三", "哪些")
    )
    wants_compare = len(rows) >= 3 or any(
        word in query for word in ("对比", "比较", "全景", "分别", "各家")
    )

    threshold_match = re.search(
        r"(超过|高于|不低于|大于等于|>=|≥|低于|小于|不高于|<=|≤)\s*(\d+(?:\.\d+)?)\s*%",
        query,
    )
    if threshold_match and len(metrics) == 1:
        operator, threshold_text = threshold_match.groups()
        threshold = float(threshold_text)
        if operator in ("超过", "高于", "大于等于", ">=", "≥"):
            filtered = [row for row in rows if row[metrics[0]] > threshold]
            relation = "超过"
        elif operator in ("不低于",):
            filtered = [row for row in rows if row[metrics[0]] >= threshold]
            relation = "不低于"
        elif operator in ("低于", "小于", "<=", "≤"):
            filtered = [row for row in rows if row[metrics[0]] < threshold]
            relation = "低于"
        else:
            filtered = [row for row in rows if row[metrics[0]] <= threshold]
            relation = "不高于"
        filtered.sort(key=lambda row: row[metrics[0]], reverse=True)
        if filtered:
            direct = (
                f"{METRIC_LABELS[metrics[0]]}{relation}{threshold:g}%的公司有："
                + "、".join(
                    f"{row['company']}（{_format_metric(metrics[0], row[metrics[0]])}）"
                    for row in filtered
                )
                + "。"
            )
            rows = filtered
        else:
            direct = f"样本中没有公司的{METRIC_LABELS[metrics[0]]}{relation}{threshold:g}%。"
    elif wants_rank and len(rows) > 1:
        pieces = []
        for metric in metrics:
            ordered = sorted(rows, key=lambda row: row[metric], reverse=True)
            top = ordered[:3]
            rendered = "、".join(
                f"{row['company']}（{_format_metric(metric, row[metric])}）"
                for row in top
            )
            pieces.append(f"{METRIC_LABELS[metric]}前三为{rendered}")
        direct = "；".join(pieces) + "。"
    elif wants_compare:
        lines = []
        for row in sorted(rows, key=lambda item: item["revenue_yi"], reverse=True):
            values = "，".join(
                f"{METRIC_LABELS[metric]} {_format_metric(metric, row[metric])}"
                for metric in metrics
            )
            lines.append(f"- {row['company']}：{values}")
        direct = "\n".join(lines)
    else:
        parts = []
        for row in rows:
            values = "，".join(
                f"{METRIC_LABELS[metric]}为{_format_metric(metric, row[metric])}"
                for metric in metrics
            )
            parts.append(f"{row['company']}2025年{values}")
        direct = "；".join(parts) + "。"

    citations = [
        _metric_citation(row, company_map)
        for row in rows
    ]
    return {
        "answer": direct,
        "citations": citations,
        "mode": "structured_metric",
    }


def _sentence_score(sentence: str, query: str) -> float:
    query_tokens = set(tokenize(query))
    sentence_tokens = set(tokenize(sentence))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & sentence_tokens) / len(query_tokens)
    length_penalty = 0.0 if 25 <= len(sentence) <= 280 else 0.08
    numeric_bonus = 0.0
    if any(word in query for word in ("多少", "占比", "同比", "增长率", "市占率")):
        if re.search(r"\d", sentence) and any(symbol in sentence for symbol in ("%", "元", "亿", "万")):
            numeric_bonus += 0.3
    normalized_query = query.replace("市占率", "市场占有率").replace("经营现金流", "经营活动现金流")
    query_parts = [
        part
        for part in jieba.lcut(normalized_query)
        if len(part) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", part)
    ]
    phrases = set(query_parts)
    for index in range(len(query_parts) - 1):
        combined = query_parts[index] + query_parts[index + 1]
        phrases.add(combined)
        if len(query_parts[index + 1]) >= 2:
            phrases.add(query_parts[index] + query_parts[index + 1][:2])
    phrases = {
        phrase
        for phrase in phrases
        if len(phrase) >= 3
        and not any(name in phrase for name in KNOWN_COMPANY_NAMES)
    }
    phrase_bonus = 0.0
    for phrase in phrases:
        if phrase in sentence:
            phrase_bonus += 0.22
    reason_bonus = 0.0
    if any(word in query for word in ("原因", "为什么", "驱动", "增长")):
        if any(
            phrase in sentence
            for phrase in ("同比增长", "同比增加", "主要系", "主要原因", "变动原因")
        ):
            reason_bonus += 0.35
        if any(
            phrase in sentence
            for phrase in ("营业总收入", "营业收入", "归母净利润", "主营业务", "销售规模", "核心业务")
        ):
            reason_bonus += 0.35
        if any(
            phrase in sentence
            for phrase in ("投资活动产生的现金流量", "筹资活动产生的现金流量", "现金及现金等价物净增加额")
        ):
            reason_bonus -= 0.55
    return overlap + reason_bonus + numeric_bonus + phrase_bonus - length_penalty


def answer_from_evidence(query: str, hits: list[dict]) -> dict:
    candidates = []
    for hit in hits:
        sentences = split_sentences(normalize_text(hit["text"]))
        for sentence in sentences:
            if not 12 <= len(sentence) <= 320:
                continue
            sentence_score = _sentence_score(sentence, query)
            if sentence_score <= 0.04:
                continue
            candidates.append(
                (
                    sentence_score + min(hit["score"] * 8, 0.18),
                    hit,
                    sentence,
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    matched = detect_companies(
        query,
        CONFIG_COMPANIES,
    )
    max_per_company = 2 if len(matched) <= 1 else 1
    selected = []
    company_counts: dict[str, int] = {}
    seen_sentences = set()
    for _, hit, sentence in candidates:
        signature = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", sentence)
        if signature in seen_sentences:
            continue
        if company_counts.get(hit["company"], 0) >= max_per_company:
            continue
        selected.append((hit, sentence))
        seen_sentences.add(signature)
        company_counts[hit["company"]] = company_counts.get(hit["company"], 0) + 1
        if len(selected) >= (4 if max_per_company == 1 else 3):
            break

    citations = []
    quotes = []
    for hit, sentence in selected:
        quotes.append(f"{hit['company']}：{sentence}")
        citations.append(
            {
                "company": hit["company"],
                "code": hit["code"],
                "report_title": hit["report_title"],
                "page": hit["page"],
                "chapter": hit["chapter"],
                "source_url": hit["source_url"],
                "pdf_url": hit["pdf_url"],
                "kind": "text_evidence",
            }
        )

    answer = "根据年报原文，检索到的直接依据如下：\n\n" + "\n\n".join(
        f"- {quote}" for quote in quotes
    )
    return {
        "answer": answer,
        "citations": citations,
        "mode": "extractive",
    }


def ask(
    retriever: HybridRetriever,
    query: str,
    selected_companies: list[str] | None = None,
) -> dict:
    companies = json.loads((ROOT / "config" / "companies.json").read_text(encoding="utf-8"))
    metrics_rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    matched = detect_companies(query, companies)
    if selected_companies:
        matched = [company for company in matched if company["name"] in selected_companies]
        if not matched:
            matched = [
                company for company in companies if company["name"] in selected_companies
            ]

    metric_names = detect_metrics(query)
    cross_company = len(matched) >= 3 or any(
        word in query for word in ("对比", "比较", "排名", "全景", "分别", "各家")
    )
    names = [company["name"] for company in matched] or selected_companies or None
    retrieval_query = query
    if any(word in query for word in ("原因", "为什么", "驱动", "增长")):
        retrieval_query += " 主要原因 同比变动 主营业务分析 经营情况"
    if any(word in query for word in ("产品", "业务", "收入构成")):
        retrieval_query += " 产品构成 营业收入构成 分产品"
    if any(word in query for word in ("渠道", "销售模式", "经销", "直销")):
        retrieval_query += " 分销售模式 渠道"
    if any(word in query for word in ("海外", "境外", "出海")):
        retrieval_query += " 境外收入 海外业务"
    hits = retriever.search(
        retrieval_query,
        companies=names,
        top_k=12 if cross_company else 8,
        per_company=cross_company,
    )

    if matched and metric_names:
        metric_result = answer_metrics(
            query,
            metric_names,
            matched,
            metrics_rows,
            manifest,
        )
        if any(word in query for word in ("原因", "为什么", "说明")):
            evidence_result = answer_from_evidence(query, hits)
            evidence_text = evidence_result["answer"].split("\n\n", 1)[-1]
            result = {
                "answer": (
                    metric_result["answer"]
                    + "\n\n原因检索结果：\n\n"
                    + evidence_text
                ),
                "citations": metric_result["citations"] + evidence_result["citations"],
                "mode": "structured_metric + extractive_reason",
            }
        else:
            result = metric_result
    else:
        result = answer_from_evidence(query, hits)
    result["chunks"] = hits
    result["matched_companies"] = [company["name"] for company in matched]
    return result

