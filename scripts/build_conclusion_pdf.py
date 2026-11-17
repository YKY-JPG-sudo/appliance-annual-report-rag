#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conclusion.pdf"


def main() -> None:
    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    pdfmetrics.registerFont(TTFont("CN", font_path))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="家电年报问答知识库：一页结论",
        author="Codex",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName="CN",
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#18251E"),
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    kicker = ParagraphStyle(
        "Kicker",
        fontName="CN",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#D25F38"),
        spaceAfter=4,
    )
    heading = ParagraphStyle(
        "Heading",
        fontName="CN",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#18251E"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        fontName="CN",
        fontSize=9.2,
        leading=14,
        textColor=colors.HexColor("#2D3730"),
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small",
        fontName="CN",
        fontSize=7.3,
        leading=10,
        textColor=colors.HexColor("#68746C"),
    )

    metric_data = [
        ["12 家公司", "2,501 页", "3,900 张表", "6,409 块", "9 / 10"],
        ["年报样本", "全文页数", "行列还原表", "检索块", "自动判分"],
    ]
    metric_table = Table(metric_data, colWidths=[29 * mm] * 5, rowHeights=[13 * mm, 7 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#18251E")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "CN"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("FONTSIZE", (0, 1), (-1, 1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#405248")),
            ]
        )
    )

    story = [
        Paragraph("FINANCIAL REPORT RAG · 2025", kicker),
        Paragraph("家电年报问答知识库：一页结论", title),
        metric_table,
        Spacer(1, 5 * mm),
        Paragraph("核心判断", heading),
        Paragraph(
            "12 家 A 股家电公司的 2025 年年报已建成“文字 + 表格 + 页码”知识库，"
            "采用中文 BGE 向量与 BM25 双路召回、Reciprocal Rank Fusion 融合。"
            "固定 10 题自动判分 9/10；每题均保留召回块、页码、引用与对错记录。",
            body,
        ),
        Paragraph("哪类题答得好", heading),
        Paragraph(
            "<b>• 单点事实与表格行最稳定。</b>营收、归母净利润、经营现金流、同比增长，"
            "以及境外销售、线下市占率等表格数值，能够回到具体年报页并给出单位与同比。",
            body,
        ),
        Paragraph(
            "<b>• 跨公司排名和单指标阈值筛选也稳定。</b>结构化指标先统一为亿元和百分比，"
            "再执行排序或阈值过滤，避免不同公司的“元/千元”口径混用。",
            body,
        ),
        Paragraph(
            "<b>• 数字加原因适合混合回答。</b>先返回财务指标，再从管理层讨论中抽取原因；"
            "飞科经营现金流 +195.42% 及“支付采购款和营销广告费减少”均被正确引用。",
            body,
        ),
        Paragraph("哪类题翻车", heading),
        Paragraph(
            "<b>复合筛选失败。</b>“营业收入增长但归母净利润下降”应只返回苏泊尔"
            "（营收 +1.54%，归母净利润 -6.58%）。系统却分别列出营收与利润 Top-3，"
            "没有执行两个条件的交集。",
            body,
        ),
        Paragraph("为什么", heading),
        Paragraph(
            "检索并非主要瓶颈：苏泊尔第 7 页已在 Top-10 中被召回，引用也正确。"
            "问题出在查询规划：指标识别器能看到两个指标，却只有“取前三/超过阈值”两类操作，"
            "缺少可组合的比较表达式。另一个结构性风险来自 PDF 合并单元格、跨页表头，"
            "会造成表格首行标签错位；当前以保留原始行列来支持人工复核。",
            body,
        ),
        KeepTogether(
            [
                Paragraph("下一步", heading),
                Paragraph(
                    "把问题解析为可执行查询树（公司集合、指标、比较符、AND/OR），先过滤结构化指标，"
                    "再让向量/BM25 负责原因证据；同时补跨页表头与指标类型重排。这样可把"
                    "“文本检索”升级为“检索 + 可审计计算”。",
                    body,
                ),
            ]
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "来源：巨潮资讯网 2025 年年度报告。逐题记录见 outputs/evaluation.md；"
            "原始链接与 SHA-256 见 data/manifest.json。本文仅用于年报研究，不构成投资建议。",
            small,
        ),
    ]

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8DED8"))
        canvas.line(18 * mm, 10 * mm, 192 * mm, 10 * mm)
        canvas.setFont("CN", 7)
        canvas.setFillColor(colors.HexColor("#79857D"))
        canvas.drawString(18 * mm, 6.4 * mm, "家电年报问答库 · 12 家上市公司 · 2025 年")
        canvas.drawRightString(192 * mm, 6.4 * mm, "1 / 1")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer)
    print(OUT)


if __name__ == "__main__":
    main()
