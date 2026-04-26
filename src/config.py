"""
Central configuration for the CC Benefits RAG pipeline.
All tunable constants live here — don't scatter magic values across modules.
"""

from pathlib import Path

# ── Storage ───────────────────────────────────────────────────────────────────

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "cc_benefits"

# ── Models ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "rerank-v3.5"  # Cohere

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K = 8          # candidates fetched per card from each retriever (vector + BM25)
RERANK_TOP_K = 5   # max chunks kept after reranking
RERANK_MIN_K = 2   # min chunks always kept (floor to protect recall)
SCORE_GAP_THRESHOLD = 0.35  # drop remaining chunks when consecutive score gap exceeds this
RRF_K = 60         # reciprocal rank fusion constant (higher = smoother rank blending)

# ── Cards ─────────────────────────────────────────────────────────────────────

# Canonical names — must match what index.py stores in metadata
CARD_NAMES = [
    "Amex Gold",
    "Amex Platinum",
    "Amex Delta Gold",
    "Bilt Palladium",
    "Capital One Venture X",
    "United Explorer",
]

# ── LLM system prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a credit card benefits assistant.

Answer ONLY using the provided context from the benefits documents.

IMPORTANT RULES:
- If the context does not mention a benefit, say clearly:
  "This benefit is not mentioned in the [Card Name] benefits guide."
- Never assume a benefit exists if it is not explicitly stated in the context.
- Never answer from general knowledge about credit cards.
- Always state which card you are referring to."""
