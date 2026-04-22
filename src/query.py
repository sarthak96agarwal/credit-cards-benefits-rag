"""
Query orchestration.

Ties together card detection, retrieval, and response synthesis:
  1. detect_cards  — identify which card(s) the question is about
  2. hybrid_retrieve — vector + BM25 + RRF + rerank, per card
  3. synthesize    — generate a grounded answer from the retrieved chunks
"""

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import EMBED_MODEL, LLM_MODEL, RERANK_TOP_K, SYSTEM_PROMPT
from card_detector import detect_cards
from store import load_index, get_all_nodes
from retriever import hybrid_retrieve

load_dotenv()

Settings.llm = OpenAI(model=LLM_MODEL, temperature=0, system_prompt=SYSTEM_PROMPT)
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)


# ── Engine loader ─────────────────────────────────────────────────────────────

def load_query_engine() -> tuple:
    """
    Load the index and prepare all nodes for BM25.

    Returns:
        (index, all_nodes) — pass this to query() to avoid reloading on each call.
    """
    index, collection = load_index()
    all_nodes = get_all_nodes(collection)
    return index, all_nodes


# ── Main query function ───────────────────────────────────────────────────────

def query(question: str, query_engine: tuple | None = None) -> dict:
    """
    Ask a question and return a grounded answer with source attribution.

    Card detection determines retrieval scope:
    - 0 cards detected  → unfiltered search across all cards
    - 1 card detected   → search filtered to that card only
    - 2+ cards detected → per-card search, results merged

    Args:
        question:     The user's natural-language question.
        query_engine: (index, all_nodes) from load_query_engine(). If None,
                      loads fresh from disk (slower — use for one-off calls).

    Returns:
        {
            "answer":         str,
            "sources":        list of {text, card_name, source_file, page, score},
            "detected_cards": list of card names found in the question,
        }
    """
    index, all_nodes = query_engine if query_engine else load_query_engine()
    detected = detect_cards(question)

    if len(detected) == 0:
        nodes = hybrid_retrieve(index, all_nodes, question)

    elif len(detected) == 1:
        nodes = hybrid_retrieve(index, all_nodes, question, card_name=detected[0])

    else:
        # Per-card retrieval — keeps each card's chunks from being diluted
        # by the other card's semantic similarity scores during vector search.
        merged = []
        for card in detected:
            merged.extend(hybrid_retrieve(index, all_nodes, question, card_name=card))
        # Re-sort by reranker score; keep top N per card
        merged.sort(key=lambda n: n.score or 0, reverse=True)
        nodes = merged[:RERANK_TOP_K * len(detected)]

    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(question, nodes=nodes)

    sources = [
        {
            "text": node.node.text,
            "card_name": node.node.metadata.get("card_name", "Unknown"),
            "source_file": node.node.metadata.get("source", ""),
            "page": node.node.metadata.get("page_label", "?"),
            "score": round(node.score, 4) if node.score else None,
        }
        for node in response.source_nodes
    ]

    return {
        "answer": str(response),
        "sources": sources,
        "detected_cards": detected,
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    reranker_status = "enabled" if os.getenv("COHERE_API_KEY") else "disabled (set COHERE_API_KEY to enable)"
    print(f"\n💳 Credit Card Benefits RAG — Phase 2 (Hybrid + Reranker {reranker_status})\n")

    engine = load_query_engine()

    test_questions = [
        "What is the auto rental coverage on the United Explorer Card?",
        "Does the Amex Gold card include airport lounge access?",
        "How does the Trip Delay differ between the Capital One Venture X and the United Explorer?",
        "Which card has a higher Trip Cancellation limit, the United Explorer Card or the Bilt Palladium?",
    ]

    for q in test_questions:
        result = query(q, engine)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"   Detected: {result['detected_cards']}")
        print(f"   Sources:  {[s['card_name'] for s in result['sources']]}")
        print(f"   Scores:   {[s['score'] for s in result['sources']]}")
        print()
