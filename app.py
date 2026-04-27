"""
Streamlit UI for the Handbook QA System.

Run:  streamlit run app.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from ingestion import ingest
from tfidf_baseline import TFIDFRetriever
from retriever import HybridRetriever
from answer_gen import generate_answer
from extension import find_query_patterns

st.set_page_config(page_title="NUST Handbook QA", layout="wide")
st.title("📚 NUST Handbook QA System")
st.caption("Big Data project — Hybrid LSH (MinHash + SimHash) vs TF-IDF baseline + LLM answers")


@st.cache_resource(show_spinner=False)
def load_system(pdf_path: str, num_perm: int, lsh_threshold: float,
                simhash_bits: int, simhash_threshold: int):
    chunks = ingest(pdf_path)
    hybrid = HybridRetriever(
        num_perm=num_perm,
        lsh_threshold=lsh_threshold,
        simhash_bits=simhash_bits,
        simhash_threshold=simhash_threshold,
    )
    hybrid.build(chunks)
    tfidf = TFIDFRetriever()
    tfidf.build(chunks)
    return chunks, hybrid, tfidf


# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    pdf_path = st.text_input("Handbook PDF path", "data/handbook.pdf")

    st.subheader("Retrieval")
    method = st.radio("Method", ["Compare both", "Hybrid LSH", "TF-IDF (baseline)"])
    top_k = st.slider("Top-k chunks", 1, 10, 5)

    st.subheader("LSH params")
    num_perm = st.select_slider("MinHash num_perm", [32, 64, 128, 256], 128)
    lsh_threshold = st.slider("LSH Jaccard threshold", 0.05, 0.5, 0.2, 0.05)
    simhash_bits = st.select_slider("SimHash bits", [32, 64, 128], 64)
    simhash_threshold = st.slider("SimHash Hamming threshold", 8, 40, 28)

    st.subheader("LLM (optional)")
    use_llm = st.checkbox("Use LLM for answers", value=True)
    api_key = st.text_input("API key", type="password",
                            help="OpenAI / Groq / OpenRouter compatible")
    base_url = st.text_input("Base URL", "https://api.openai.com/v1")
    model = st.text_input("Model", "gpt-4o-mini")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


if not Path(pdf_path).exists():
    st.error(f"❌ PDF not found at `{pdf_path}`. Place handbook there and reload.")
    st.stop()

with st.spinner("Loading & indexing handbook..."):
    chunks, hybrid, tfidf = load_system(
        pdf_path, num_perm, lsh_threshold, simhash_bits, simhash_threshold
    )

st.success(f"✅ Loaded **{len(chunks)} chunks** across "
           f"{len(set(c['page'] for c in chunks))} pages.")

if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.markdown("---")
query = st.text_input(
    "❓ Ask a question:",
    "What is the minimum GPA requirement?",
)
go = st.button("🔍 Search", type="primary")


def render_results(label, results, t_ms, color):
    st.markdown(f"### {color} {label}")
    st.caption(f"⏱️ Retrieval: {t_ms:.1f} ms · {len(results)} chunks")
    for i, r in enumerate(results, 1):
        with st.expander(f"Chunk {i} — Page {r['page']} — score {r['score']:.3f}"):
            st.write(r["text"])
            extras = []
            if "mh_score" in r:
                extras.append(f"MinHash: {r['mh_score']:.3f}")
            if "sh_score" in r:
                extras.append(f"SimHash: {r['sh_score']:.3f}")
            if "hamming" in r:
                extras.append(f"Hamming: {r['hamming']}")
            if extras:
                st.caption(" · ".join(extras))


if go and query:
    st.session_state.query_log.append(query)

    if method == "Compare both":
        c1, c2 = st.columns(2)
        with c1:
            t0 = time.time()
            r1 = hybrid.query(query, top_k=top_k)
            t1 = (time.time() - t0) * 1000
            render_results("Hybrid LSH (MinHash + SimHash)", r1, t1, "🔵")
            a1 = generate_answer(query, r1, use_llm=use_llm,
                                 api_key=api_key, model=model, base_url=base_url)
            st.info(f"**Answer ({a1['method']}):** {a1['answer']}")
        with c2:
            t0 = time.time()
            r2 = tfidf.query(query, top_k=top_k)
            t2 = (time.time() - t0) * 1000
            render_results("TF-IDF baseline (exact)", r2, t2, "🟠")
            a2 = generate_answer(query, r2, use_llm=use_llm,
                                 api_key=api_key, model=model, base_url=base_url)
            st.info(f"**Answer ({a2['method']}):** {a2['answer']}")

        st.markdown("---")
        st.caption(f"Speedup vs TF-IDF: {t2/max(t1,1e-6):.2f}× "
                   f"(LSH = {t1:.1f}ms, TF-IDF = {t2:.1f}ms)")
    else:
        retriever = hybrid if method == "Hybrid LSH" else tfidf
        t0 = time.time()
        results = retriever.query(query, top_k=top_k)
        t = (time.time() - t0) * 1000
        ans = generate_answer(query, results, use_llm=use_llm,
                              api_key=api_key, model=model, base_url=base_url)
        st.markdown(f"## 📝 Answer  *(via {ans['method']})*")
        st.info(ans["answer"])
        render_results(method, results, t, "🔵" if "LSH" in method else "🟠")


# Extension: query patterns
st.markdown("---")
with st.expander("📊 Extension — Frequent query patterns (Apriori / FIM)"):
    st.caption("Mines this session's query log for frequent token co-occurrences.")
    if len(st.session_state.query_log) >= 3:
        patterns = find_query_patterns(st.session_state.query_log, min_support=2, max_k=3)
        shown = 0
        for items, sup in patterns:
            if len(items) >= 2:
                st.write(f"`{', '.join(sorted(items))}`  →  appears in **{sup}** queries")
                shown += 1
                if shown >= 15:
                    break
        if shown == 0:
            st.write("No multi-term frequent itemsets yet.")
    else:
        st.write(f"Submit at least 3 queries to mine patterns "
                 f"({len(st.session_state.query_log)}/3).")
    if st.session_state.query_log:
        st.markdown("**Query log:**")
        for i, q in enumerate(st.session_state.query_log, 1):
            st.write(f"{i}. {q}")
