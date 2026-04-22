"""
Streamlit UI for the credit card benefits RAG chatbot.
Run with: streamlit run src/app.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from query import load_query_engine, query
from config import CARD_NAMES, TOP_K, RERANK_TOP_K, EMBED_MODEL, LLM_MODEL, RERANK_MODEL

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="💳 Card Benefits Assistant",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Credit Card Benefits Assistant")
reranker_on = bool(os.getenv("COHERE_API_KEY"))
phase_note = "Hybrid Search + Reranker" if reranker_on else "Hybrid Search (reranker disabled)"
st.caption(f"Phase 2 — {phase_note}")

# ── Load engine (cached so it doesn't reload on every interaction) ─────────────

@st.cache_resource
def get_engine():
    try:
        return load_query_engine()
    except FileNotFoundError:
        return None

engine = get_engine()

if engine is None:
    st.error("⚠️ No index found. Run `python src/index.py` first, then refresh this page.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            if meta.get("detected_cards"):
                st.caption(f"🃏 Detected cards: {', '.join(meta['detected_cards'])}")
            with st.expander("📄 Retrieved chunks"):
                for i, src in enumerate(meta["sources"]):
                    st.markdown(f"**[{i+1}] {src['card_name']}** — Page {src['page']} | Score: `{src['score']}`")
                    st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your card benefits..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching benefits docs..."):
            result = query(prompt, engine)

        st.write(result["answer"])

        if result["detected_cards"]:
            st.caption(f"🃏 Detected cards: {', '.join(result['detected_cards'])}")

        with st.expander("📄 Retrieved chunks"):
            if not result["sources"]:
                st.write("No chunks retrieved.")
            for i, src in enumerate(result["sources"]):
                st.markdown(f"**[{i+1}] {src['card_name']}** — Page {src['page']} | Score: `{src['score']}`")
                st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {"detected_cards": result["detected_cards"], "sources": result["sources"]},
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    st.markdown(f"""
    **Retrieval:** Hybrid (vector + BM25 + RRF)
    **Reranker:** {"✅ Cohere " + RERANK_MODEL if reranker_on else "❌ disabled"}
    **Candidates per card:** {TOP_K} × 2 retrievers → RRF
    **Final chunks:** {RERANK_TOP_K} (after rerank)
    **LLM:** {LLM_MODEL}
    **Embeddings:** {EMBED_MODEL}
    """)
    st.divider()
    st.markdown("**Supported cards:**")
    for card in CARD_NAMES:
        st.markdown(f"- {card}")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
