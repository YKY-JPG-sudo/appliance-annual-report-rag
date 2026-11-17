#!/usr/bin/env python3
"""Download the 2025 annual reports from CNINFO and write a source manifest."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ROOT / "config" / "companies.json"
REPORT_DIR = ROOT / "data" / "reports"
MANIFEST = ROOT / "data" / "manifest.json"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_BASE = "https://static.cninfo.com.cn/"
DETAIL_BASE = "https://www.cninfo.com.cn/new/disclosure/detail"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
    ),
    "Referer": "https://www.cninfo.com.cn/",
    "X-Requested-With": "XMLHttpRequest",
}


def clean_title(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def find_report(company: dict) -> dict:
    payload = {
        "pageNum": "1",
        "pageSize": "30",
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{company['code']},{company['org_id']}",
        "searchkey": "2025年年度报告",
        "seDate": "2025-01-01~2026-11-18",
        "tdate": "",
        "type": "",
        "isHLtitle": "true",
    }
    response = requests.post(
        QUERY_URL,
        data=payload,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    records = response.json().get("announcements") or []

    candidates = []
    for record in records:
        title = clean_title(record.get("announcementTitle", ""))
        if "2025年年度报告" not in title:
            continue
        if any(marker in title for marker in ("摘要", "英文", "更正", "已取消", "取消")):
            continue
        candidates.append((title, record))

    if not candidates:
        raise RuntimeError(f"No Chinese 2025 annual report found for {company['name']}")

    # Prefer an exact annual report title; otherwise use the shortest matching title.
    candidates.sort(
        key=lambda item: (
            0 if item[0].endswith("2025年年度报告") else 1,
            len(item[0]),
            -int(item[1].get("announcementTime") or 0),
        )
    )
    title, record = candidates[0]
    announcement_id = str(record["announcementId"])
    date = datetime.fromtimestamp(
        int(record["announcementTime"]) / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")
    detail_url = (
        f"{DETAIL_BASE}?stockCode={company['code']}"
        f"&announcementId={announcement_id}"
        f"&orgId={company['org_id']}&announcementTime={date}"
    )
    pdf_url = PDF_BASE + str(record["adjunctUrl"]).lstrip("/")
    return {
        **company,
        "report_title": title,
        "announcement_id": announcement_id,
        "announcement_date": date,
        "detail_url": detail_url,
        "pdf_url": pdf_url,
        "reported_size_kb": record.get("adjunctSize"),
    }


def download(url: str, target: Path) -> tuple[int, str]:
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {url}")
    target.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    companies = json.loads(COMPANIES.read_text(encoding="utf-8"))
    manifest = []

    for index, company in enumerate(companies, start=1):
        print(f"[{index:02d}/{len(companies)}] 检索 {company['name']}...", flush=True)
        item = find_report(company)
        target = REPORT_DIR / f"{company['code']}_{company['name']}_2025.pdf"
        size, digest = download(item["pdf_url"], target)
        item.update(
            {
                "local_path": str(target.relative_to(ROOT)),
                "bytes": size,
                "sha256": digest,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "report_period": "2025",
            }
        )
        manifest.append(item)
        print(f"    {item['report_title']} -> {target.name} ({size / 1024:.0f} KiB)")

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"下载失败: {exc}", file=sys.stderr)
        raise

