from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from src.config import DATA_DIR, ROOT
from src.rag import HybridRetriever, ask


st.set_page_config(
    page_title="家电年报问答库",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#17211b; --muted:#647068; --accent:#d25f38; --paper:#f7f4ed; }
    .stApp { background: linear-gradient(135deg, #f8f5ee 0%, #eef3eb 100%); color: var(--ink); }
    [data-testid="stSidebar"] { background:#18251e; color:#edf3ec; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color:#edf3ec !important; }
    .hero { background:#18251e; color:#f7f4ed; padding:34px 38px; border-radius:22px;
            box-shadow:0 18px 50px rgba(24,37,30,.15); margin-bottom:20px; }
    .hero .eyebrow { color:#ef9b76; font-size:.82rem; font-weight:700; letter-spacing:.16em; }
    .hero h1 { font-family:Georgia,"Songti SC",serif; font-size:2.55rem; margin:.35rem 0 .55rem; }
    .hero p { color:#cad7ce; font-size:1.02rem; max-width:820px; line-height:1.7; margin:0; }
    .metric-row { display:flex; gap:12px; flex-wrap:wrap; margin-top:18px; }
    .metric-chip { border:1px solid rgba(255,255,255,.17); background:rgba(255,255,255,.07);
                   padding:8px 13px; border-radius:999px; font-size:.88rem; }
    .answer-card { background:rgba(255,255,255,.88); border:1px solid #dfe5dc; border-left:5px solid var(--accent);
                   padding:22px 25px; border-radius:16px; box-shadow:0 8px 25px rgba(33,43,36,.06); }
    .answer-card h3 { margin:0 0 10px; color:var(--ink); font-family:Georgia,"Songti SC",serif; }
    .answer-card p, .answer-card li { color:#2d3730; line-height:1.8; }
    .source-card { background:rgba(255,255,255,.78); border:1px solid #e1e6df; padding:15px 17px;
                   border-radius:13px; margin:9px 0; }
    .source-kicker { font-size:.76rem; color:var(--accent); font-weight:800; letter-spacing:.08em; }
    .source-meta { color:var(--muted); font-size:.86rem; margin:.35rem 0 .55rem; }
    .source-quote { color:#334038; font-size:.91rem; line-height:1.65; }
    .stTextInput input { border-radius:12px !important; border:1px solid #cfd8cf !important; padding:14px 16px !important; }
    .stButton button { background:var(--accent) !important; color:white !important; border:0 !important;
                       border-radius:12px !important; min-height:46px !important; font-weight:700 !important; }
    .small-note { color:#758178; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载年报索引...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
build = json.loads((DATA_DIR / "build_summary.json").read_text(encoding="utf-8"))
retriever = get_retriever()

with st.sidebar:
    st.markdown("## 年报知识库")
    st.caption("2025 年度报告 · 巨潮资讯")
    st.markdown(
        f"**{build['companies']} 家公司**  \n"
        f"**{build['pages']:,} 页**  \n"
        f"**{build['tables']:,} 张表**  \n"
        f"**{build['chunks']:,} 个检索块**"
    )
    st.divider()
    selected_companies = st.multiselect(
        "限定公司（可不选）",
        options=retriever.companies,
        placeholder="全部公司",
    )
    st.divider()
    st.markdown("**索引**")
    st.caption("中文 BGE dense vector + BM25，经 Reciprocal Rank Fusion 融合。")
    st.markdown("**出处**")
    st.caption("每个答案显示公司、章节、PDF 页码和巨潮公告链接。")
    st.divider()
    with st.expander("报告清单"):
        for item in manifest:
            st.markdown(
                f"[{item['name']} {item['code']}]({item['detail_url']})  \n"
                f"<span class='small-note'>年报页码 {item.get('pdf_pages', '')}</span>",
                unsafe_allow_html=True,
            )

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">FINANCIAL REPORT RAG</div>
      <h1>家电上市公司年报问答</h1>
      <p>面向 12 家 A 股家电公司的 2025 年年度报告。问题可跨公司比较，
      答案优先引用结构化财务指标；叙述性问题回到带页码的年报原文。</p>
      <div class="metric-row">
        <span class="metric-chip">{build['companies']} 家公司</span>
        <span class="metric-chip">{build['pages']:,} 页原文</span>
        <span class="metric-chip">{build['tables']:,} 张行列还原表</span>
        <span class="metric-chip">向量 + BM25</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

suggestions = [
    "12家公司中，2025年营业收入和归母净利润最高的分别是谁？",
    "哪些公司经营现金流同比增长率超过50%？",
    "海尔智家2025年营业收入、归母净利润和经营现金流分别是多少？",
    "美的集团业绩增长的主要原因是什么？",
]

if "question" not in st.session_state:
    st.session_state.question = suggestions[0]

cols = st.columns(len(suggestions))
for col, suggestion in zip(cols, suggestions):
    if col.button(suggestion, use_container_width=True):
        st.session_state.question = suggestion

query = st.text_input(
    "向年报提问",
    key="question",
    placeholder="例如：12家公司谁的净利润增速最高？",
    label_visibility="collapsed",
)
ask_clicked = st.button("检索并回答", type="primary", use_container_width=True)

if ask_clicked and query.strip():
    with st.spinner("检索年报中..."):
        result = ask(retriever, query.strip(), selected_companies)

    st.markdown(
        f"""
        <div class="answer-card">
          <h3>结论</h3>
          <p>{result['answer'].replace(chr(10), '<br>')}</p>
          <div class="small-note">回答模式：{result['mode']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        setTimeout(() => {
          const card = window.parent.document.querySelector(".answer-card");
          if (card) card.scrollIntoView({behavior: "smooth", block: "start"});
        }, 120);
        </script>
        """,
        height=0,
    )

    st.subheader("答案出处")
    for index, citation in enumerate(result["citations"], start=1):
        st.markdown(
            f"""
            <div class="source-card">
              <div class="source-kicker">SOURCE {index:02d}</div>
              <div class="source-meta">{citation['company']} · {citation['code']} ·
              {citation['report_title']} · 第 {citation['page']} 页 · {citation['chapter']}</div>
              <a href="{citation['source_url']}" target="_blank">打开巨潮公告页</a>
              &nbsp;·&nbsp; <a href="{citation['pdf_url']}" target="_blank">打开 PDF</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f"查看召回的 {len(result['chunks'])} 个原始块"):
        for rank, chunk in enumerate(result["chunks"], start=1):
            st.markdown(
                f"**#{rank} {chunk['company']} · 第 {chunk['page']} 页 · "
                f"{chunk['chapter']}**  \n"
                f"`chunk_id={chunk['chunk_id']}`  "
                f"`RRF={chunk['score']}`  `dense={chunk['dense_score']}`  "
                f"`BM25={chunk['bm25_score']}`"
            )
            st.caption(chunk["text"][:500].replace("\n", " "))

if not ask_clicked:
    st.markdown("### 可问什么")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**单公司事实**  \n"
            "营收、利润、现金流、同比变化、研发投入、业务与渠道。"
        )
    with right:
        st.markdown(
            "**跨公司全景**  \n"
            "规模排名、增速排名、现金流方向、利润与收入是否背离。"
        )

st.divider()
st.caption(
    "数据口径：各公司 2025 年年度报告；除特别说明外，财务指标为合并报表口径。"
    " 本页用于年报检索与研究，不构成投资建议。"
)
