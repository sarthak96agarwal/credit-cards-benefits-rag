"""
Retrieval pipeline: vector search + BM25 + RRF fusion + Cohere reranking.

Flow per card:
  1. vector_retrieve  — dense cosine similarity search (ChromaDB)
  2. bm25_retrieve    — keyword search (BM25S)
  3. rrf_fuse         — merge and re-rank both lists via Reciprocal Rank Fusion
  4. rerank           — Cohere cross-encoder re-scores the fused candidates
"""

import os

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever

from config import TOP_K, RERANK_TOP_K, RRF_K, RERANK_MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card_filter(card_name: str) -> MetadataFilters:
    return MetadataFilters(filters=[
        MetadataFilter(key="card_name", value=card_name, operator=FilterOperator.EQ),
    ])


# ── Step 1: Dense vector retrieval ───────────────────────────────────────────

def vector_retrieve(
    index: VectorStoreIndex,
    question: str,
    card_name: str | None = None,
    top_k: int = TOP_K,
) -> list[NodeWithScore]:
    """Retrieve top-k chunks by cosine similarity, optionally filtered to one card."""
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=_card_filter(card_name) if card_name else None,
    )
    return retriever.retrieve(question)


# ── Step 2: BM25 keyword retrieval ───────────────────────────────────────────

def bm25_retrieve(
    all_nodes: list[TextNode],
    question: str,
    card_name: str | None = None,
    top_k: int = TOP_K,
) -> list[NodeWithScore]:
    """Retrieve top-k chunks by BM25 keyword scoring, optionally filtered to one card."""
    corpus = (
        [n for n in all_nodes if n.metadata.get("card_name") == card_name]
        if card_name else all_nodes
    )
    retriever = BM25Retriever.from_defaults(nodes=corpus, similarity_top_k=top_k)
    return retriever.retrieve(question)


# ── Step 3: Reciprocal Rank Fusion ───────────────────────────────────────────

def rrf_fuse(
    vector_nodes: list[NodeWithScore],
    bm25_nodes: list[NodeWithScore],
    k: int = RRF_K,
) -> list[NodeWithScore]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Each node's score = sum of 1/(k + rank) across the lists it appears in.
    Nodes appearing in both lists get a boost; nodes in only one list still qualify.
    The constant k (default 60) controls how much top-rank positions are rewarded.
    """
    scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for rank, node in enumerate(vector_nodes):
        nid = node.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1 / (k + rank + 1)
        node_map[nid] = node

    for rank, node in enumerate(bm25_nodes):
        nid = node.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1 / (k + rank + 1)
        if nid not in node_map:
            node_map[nid] = node

    return [
        NodeWithScore(node=node_map[nid].node, score=scores[nid])
        for nid in sorted(scores, key=lambda n: scores[n], reverse=True)
    ]


# ── Step 4: Cohere reranking ─────────────────────────────────────────────────

_last_rerank_time: float = 0.0
_RERANK_MIN_INTERVAL = 7.0  # seconds between Cohere calls (trial key: 10/min)


def rerank(
    nodes: list[NodeWithScore],
    question: str,
    top_k: int = RERANK_TOP_K,
) -> list[NodeWithScore]:
    """
    Re-score nodes using Cohere's cross-encoder reranker.

    The reranker evaluates each (question, chunk) pair jointly — unlike
    embeddings which encode query and chunk independently. This catches
    cases where a chunk is superficially similar to the query (high cosine
    similarity) but doesn't actually answer it, and vice versa.

    Falls back to the top_k RRF results if COHERE_API_KEY is not set.
    Enforces a minimum interval between calls for trial key rate limits.
    """
    import time

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return nodes[:top_k]

    global _last_rerank_time
    elapsed = time.time() - _last_rerank_time
    if elapsed < _RERANK_MIN_INTERVAL:
        time.sleep(_RERANK_MIN_INTERVAL - elapsed)

    from llama_index.postprocessor.cohere_rerank import CohereRerank
    reranker = CohereRerank(api_key=api_key, model=RERANK_MODEL, top_n=top_k)
    result = reranker.postprocess_nodes(nodes, query_str=question)
    _last_rerank_time = time.time()
    return result


# ── Combined pipeline ─────────────────────────────────────────────────────────

def hybrid_retrieve(
    index: VectorStoreIndex,
    all_nodes: list[TextNode],
    question: str,
    card_name: str | None = None,
) -> list[NodeWithScore]:
    """
    Full retrieval pipeline for a single card (or unfiltered across all cards):
      vector → BM25 → RRF → rerank
    """
    v_nodes = vector_retrieve(index, question, card_name=card_name)
    b_nodes = bm25_retrieve(all_nodes, question, card_name=card_name)
    fused = rrf_fuse(v_nodes, b_nodes)
    return rerank(fused, question)
