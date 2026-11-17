# 家电上市公司年报问答知识库

面向 12 家 A 股家电公司的 2025 年年度报告，提供带页码出处的问答页面。

## 数据范围

| 公司 | 代码 | 公司 | 代码 |
| --- | --- | --- | --- |
| 美的集团 | 000333 | 格力电器 | 000651 |
| 海尔智家 | 600690 | 苏泊尔 | 002032 |
| 九阳股份 | 002242 | 老板电器 | 002508 |
| 华帝股份 | 002035 | 小熊电器 | 002959 |
| 海信视像 | 600060 | 海信家电 | 000921 |
| TCL科技 | 000100 | 飞科电器 | 603868 |

- 报告期间：2025 年
- 来源：巨潮资讯网（CNINFO）
- 原始 PDF：`data/reports/`
- 来源链接、公告日期、SHA-256：`data/manifest.json`
- 规模：2,501 页、3,900 张行列还原表、6,409 个检索块

## 实现

1. `scripts/download_reports.py` 按股票代码和巨潮 `orgId` 检索中文版年报，下载全文 PDF，并保存公告页、PDF 地址、日期、大小和校验值。
2. `scripts/build_index.py` 用 pdfplumber 逐页提取文字，使用带线框的表格解析策略还原行列，并写入 `data/tables.jsonl`。每个检索块带公司、代码、章节、PDF 页码、公告页和 PDF 链接。
3. 稠密索引使用 `BAAI/bge-small-zh-v1.5`；稀疏索引使用 BM25；两路结果经 Reciprocal Rank Fusion 融合。公司过滤、复合问法和表格短语会在检索/回答层做约束。
4. `app.py` 提供 Streamlit 问答页面。财务数值题使用经过年报页码核验的结构化指标；归因题同时返回数值和原文证据；其他问题返回抽取式原文依据。所有回答都显示公司、章节、页码和巨潮链接。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_reports.py
.venv/bin/python scripts/build_index.py
.venv/bin/streamlit run app.py
```

已经生成的 PDF 和索引可直接使用，不必重新下载或建库。若只重建索引：

```bash
.venv/bin/python scripts/build_index.py --reuse-extraction
```

## 评测

固定 10 题位于 `evaluation/questions.json`，运行：

```bash
.venv/bin/python scripts/evaluate.py
```

结果包含每题答案、命中的事实、召回 Top-K 块（含 chunk ID、页码、分数）以及引用页码：

- [逐题评测](outputs/evaluation.md)
- [完整评测 JSON](outputs/evaluation_results.json)
- [一页结论](outputs/conclusion.pdf)

本次自动判分 9/10。复合筛选题“收入增长但净利润下降”失败：当前指标回答器把“收入增长”和“利润下降”分别做 Top-3，而没有执行交集；正确答案只有苏泊尔。

## 目录

```text
app.py                       Streamlit 问答页面
config/companies.json        公司、代码和巨潮 orgId
data/reports/                原始年报 PDF
data/manifest.json           下载来源和校验记录
data/pages.jsonl             逐页文字与章节
data/tables.jsonl            行列还原后的表格
data/chunks.jsonl            带页码出处的检索块
data/financial_metrics.json  已核验的主要财务指标
indexes/                     BGE 向量、BM25 词项与统计
evaluation/                  固定的 10 道评测题
scripts/                     下载、建库、评测、截图和结论生成
outputs/                     评测、结论 PDF、页面截图
```

## 限制

- 结构化财务题只覆盖已人工核验的营收、归母净利润、经营现金流及同比，不等同于完整财务数据库。
- 合并单元格和跨页表头仍可能使表格第一行不完美；原始行列和页码会保留，便于复核。
- 复合逻辑题目前只支持单指标阈值筛选，尚未支持任意布尔表达式。
- 本工具用于年报检索与研究，不构成投资建议。

