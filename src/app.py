"""
Phase 1: Streamlit UI for the credit card benefits RAG chatbot.
Run with: streamlit run src/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from query import load_query_engine, query

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="💳 Card Benefits Assistant",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Credit Card Benefits Assistant")
st.caption("Phase 1 — Naive RAG | Ask anything about your card benefits")

# ── Load query engine (cached so it doesn't reload on every interaction) ──────

@st.cache_resource
def get_engine():
    try:
        return load_query_engine()
    except FileNotFoundError as e:
        return None

engine = get_engine()

if engine is None:
    st.error("⚠️ No index found. Run `python src/index.py` first, then refresh this page.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Retrieved chunks"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**[{i+1}] {src['card_name']}** — Page {src['page']} | Score: `{src['score']}`")
                    st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your card benefits..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching benefits docs..."):
            result = query(prompt, engine)

        st.write(result["answer"])

        with st.expander("📄 Retrieved chunks"):
            if not result["sources"]:
                st.write("No relevant chunks found above similarity threshold.")
            for i, src in enumerate(result["sources"]):
                st.markdown(f"**[{i+1}] {src['card_name']}** — Page {src['page']} | Score: `{src['score']}`")
                st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                st.divider()

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

# ── Sidebar: debug info ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Phase 1 Settings")
    st.markdown("""
    **Chunking:** Fixed 512 tokens, 50 overlap  
    **Retrieval:** Dense only (cosine similarity)  
    **Top-k:** 5 chunks  
    **Similarity cutoff:** 0.3  
    **Model:** gpt-4o-mini  
    **Embeddings:** text-embedding-3-small  
    """)
    st.divider()
    st.markdown("**Known limitations to observe:**")
    st.markdown("""
    - Chunks may cut mid-sentence  
    - Exact terms (e.g. Priority Pass) may not retrieve well  
    - No cross-card comparison support  
    - Hallucinations possible when context is weak  
    """)
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
