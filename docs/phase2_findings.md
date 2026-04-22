# Phase 2 — Metadata Filtering + Hybrid Search + Reranking: Findings & Analysis

## What We Built

Phase 2 introduced three compounding improvements on top of the Phase 1 naive RAG baseline, each targeting a specific failure mode identified in Phase 1.

### Improvement 1: LLM-Based Card Detection + Metadata Filtering

**The problem it solved:** In Phase 1, a single vector search over all 408 chunks would frequently pull chunks from the wrong card. Asking "does the Amex Gold have lounge access?" would retrieve Amex Platinum lounge chunks, causing the LLM to hallucinate "yes."

**How it works:**
- Before retrieval, a `gpt-4o-mini` call with structured JSON output identifies which card(s) the question is about.
- For single-card questions, ChromaDB's `MetadataFilter` restricts the search to only that card's chunks.
- For cross-card comparisons, separate filtered retrievals run per card and the results are merged.

**Why LLM detection instead of aliases:** A hard-coded alias dict ("Venture X" → "Capital One Venture X") breaks on paraphrasing. The LLM handles "my platinum card", "the one with the annual fee credit", and ambiguous phrasing naturally. The cost is ~$0.001/query.

### Improvement 2: Hybrid Search (Dense + BM25 + RRF)

**The problem it solved:** Pure vector search misses exact terms. A query for "Trip Cancellation limit" would rank semantically similar but wrong chunks (e.g. Trip Delay) above the correct one. Dollar amounts like "$60,000" are especially hard for embeddings.

**How it works:**
- Both a dense vector retriever (cosine similarity) and a BM25 keyword retriever run in parallel over the same card-filtered corpus.
- Results are merged with **Reciprocal Rank Fusion**: each chunk scores `1/(k + rank)` from each retriever, rewarding chunks that appear highly in both. Constant `k=60` smooths rank importance.
- This broadens the candidate pool without the noise of unfiltered retrieval.

### Improvement 3: Cohere Reranker

**The problem it solved:** After RRF, the top 5 chunks were still ranked by a blended retrieval score, not by how well each chunk actually answers the specific question. A chunk about "travel accident insurance" might rank highly because it matches "insurance" in the query, even when the question is specifically about rental car coverage.

**How it works:**
- Cohere's `rerank-v3.5` is a cross-encoder: it takes each (question, chunk) pair jointly and outputs a 0–1 relevance score.
- Unlike embeddings (which encode query and chunk independently), the cross-encoder sees both at once and can reason about whether the chunk actually answers the question.
- The top 16 RRF candidates are reranked; the top 5 by Cohere score are passed to the LLM.
- Falls back gracefully to RRF top-5 if no `COHERE_API_KEY` is set.

### Code Refactor

All logic was organized into focused modules to make the pipeline easy to extend:

| File | Responsibility |
|---|---|
| `config.py` | All constants — model names, paths, TOP_K, card names, system prompt |
| `card_detector.py` | LLM card detection |
| `store.py` | ChromaDB loading + raw node fetching for BM25 |
| `retriever.py` | `vector_retrieve`, `bm25_retrieve`, `rrf_fuse`, `rerank`, `hybrid_retrieve` |
| `query.py` | Thin orchestration: detect → retrieve → synthesize |
| `app.py` | Streamlit UI |

---

## Results

### Overall Scores

| Metric | Phase 1 | P2 Metadata | P2 Hybrid | P2 Hybrid+Rerank | Δ vs Phase 1 |
|---|---|---|---|---|---|
| Faithfulness | 0.490 | 0.625 | 0.615 | **0.644** | **+0.154** |
| Answer Relevancy | 0.653 | 0.753 | 0.747 | **0.754** | **+0.101** |
| Context Recall | 0.796 | 0.815 | 0.821 | **0.827** | **+0.031** |
| Context Precision | 0.556 | 0.570 | 0.656 | **0.658** | **+0.102** |

### Per-Category Breakdown (Phase 1 vs Final Phase 2)

| Category | Metric | Phase 1 | Phase 2 Final | Change |
|---|---|---|---|---|
| simple_fact | context_precision | 0.733 | **0.940** | +0.207 |
| simple_fact | faithfulness | 0.233 | 0.500 | +0.267 |
| exact_term | context_precision | 0.733 | **1.000** | +0.267 |
| exact_term | faithfulness | 1.000 | 0.917 | -0.083 |
| benefit_detail | faithfulness | 0.742 | **0.870** | +0.128 |
| benefit_detail | context_precision | 0.776 | 0.816 | +0.040 |
| negative | faithfulness | 0.300 | 0.500 | +0.200 |
| cross_card | context_recall | 0.333 | 0.417 | +0.084 |
| cross_card | faithfulness | 0.208 | 0.175 | -0.033 |

---

## Key Findings

### 1. Metadata filtering was the single biggest lever

Filtering retrieval to the correct card fixed the root cause of most failures: wrong-card chunks. Negative question faithfulness jumped from 0.300 → 0.700 (within the metadata-only run) because the LLM stopped seeing Amex Platinum lounge chunks when asked about Amex Gold.

### 2. The reranker fixed simple_fact and exact_term precision

In Phase 1, the right chunk was being retrieved but not ranked first — RAGAS couldn't trace the "$60,000" claim to a specific chunk when it was buried at rank 3 of 5. The reranker pushed the most directly relevant chunk to rank 1, bringing simple_fact precision from 0.733 → 0.940 and exact_term to a perfect 1.000.

### 3. Hybrid search and reranking have a tension on cross-card queries

BM25 improved cross-card recall (0.333 → 0.625) by finding exact benefit names like "Trip Cancellation" that vector search ranked too low. However, adding the reranker dropped recall back to 0.417.

The root cause: for cross-card comparisons, we retrieve 5 chunks per card (10 total), then the reranker ranks all 10 jointly. It scores the best-matching card's chunks highly and pushes the other card's chunks out of the top 5. The reranker optimizes per-chunk relevance but not per-answer coverage — it doesn't know we need chunks from *both* cards.

### 4. BM25 introduces noise on negative questions

When a benefit doesn't exist in a document, BM25 has no signal — it keyword-matches on fragments ("access", "protection") and pulls in completely unrelated chunks. This caused the hybrid-only run to regress on negative faithfulness (0.700 → 0.400). The reranker partially recovered this (→ 0.500) by filtering out the worst noise chunks.

---

## Root Causes Still Remaining

| Problem | Category | Evidence |
|---|---|---|
| Reranker drops one card in cross-card queries | cross_card | Recall 0.417; reranker ranks both cards' chunks jointly, one card wins |
| "Trip Cancellation" still partially missed | cross_card | Recall 0.417 vs 0.625 without reranker |
| Faithfulness plateau at ~0.65 | all | LLM answers are grounded but RAGAS traceability still fails on some rephrased claims |

---

## What Phase 3 Should Address

### 1. Per-card reranking for cross-card queries
Rerank each card's chunks separately before merging, guaranteeing top-N chunks from each card. The current joint reranking of 10 chunks optimizes individual chunk relevance but lets one card crowd out the other, collapsing cross_card context_precision to 0.000.

### 2. Adaptive chunk selection via reranker score gap
Instead of always keeping a fixed top-K, use the reranker's score distribution to decide how many chunks to pass to the LLM:

- Always keep a minimum of 2 chunks (floor, protects recall)
- Drop chunks where there is a large score gap between consecutive chunks (e.g. scores drop from 0.85 → 0.42 — keep 3, not 5)
- Cap at top-5

This improves precision without the recall risk of a hard threshold cutoff. A hard cutoff (e.g. score > 0.6) would drop relevant chunks whenever the reranker underscores them due to vocabulary mismatch. The gap-based approach adapts to the actual score distribution per query.

**Precision-recall tradeoff:** Any reduction in retrieved chunks risks recall if the reranker is imperfect. The score gap approach mitigates this — it only cuts when the signal is unambiguous (a sudden score cliff), rather than cutting at an arbitrary absolute threshold.

### 3. Larger chunk sizes / parent-child chunking
512 tokens frequently splits a single benefit description across two chunks (e.g. the dollar limits in chunk N, the eligibility conditions in chunk N+1). Parent-child chunking retrieves small chunks for precision but sends the full parent section to the LLM for complete context. This would improve recall on benefit_detail and cross_card without increasing noise.

### 4. Faithfulness ceiling
Even with perfect retrieval, RAGAS marks some answers unfaithful when the LLM paraphrases rather than quotes. Prompt-level fixes: instruct the LLM to copy specific amounts and terms verbatim from context, and constrain answer length to reduce filler. Structured output format (benefit → amount → conditions → source quote) would further reduce hallucinated connective language.
