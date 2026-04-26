# Phase 3 — Adaptive Retrieval: Findings & Analysis

> **Status: In Progress**

## What We're Building

Phase 3 targets the remaining gaps from Phase 2:

| Problem | Phase 2 Score | Root Cause |
|---|---|---|
| cross_card faithfulness | 0.40 | LLM gets 10 noisy chunks; one card's chunks dilute the other |
| Faithfulness plateau ~0.60 | all categories | LLM paraphrases rather than quotes; RAGAS can't trace claims |
| Negative question noise | 0.30 faithfulness | BM25 pulls irrelevant chunks; reranker only partially filters |

---

## Improvement 1: Adaptive Score Gap Cutoff

**The problem it solves:** After reranking, we always pass a fixed top-5 chunks to the LLM. For a precise factual question (e.g. "what is the Trip Cancellation limit?"), the reranker often scores the first 1-2 chunks very high and the rest much lower — but we send all 5. The trailing chunks add noise without adding information.

**How it works:**
- After Cohere reranking, scan consecutive score pairs from position `RERANK_MIN_K` downward.
- If the gap between chunk[i] and chunk[i+1] exceeds `SCORE_GAP_THRESHOLD`, cut there.
- Always keep at least `RERANK_MIN_K = 2` chunks (recall floor).
- Never exceed `RERANK_TOP_K = 5` (existing cap).

This is an adaptive cutoff, not a hard threshold — it only cuts when the signal is unambiguous (a sudden score cliff). A hard absolute threshold (e.g. score > 0.6) would drop relevant chunks whenever the reranker underscores due to vocabulary mismatch.

**Configuration (in `config.py`):**
```python
RERANK_MIN_K = 2        # floor — always keep at least this many
SCORE_GAP_THRESHOLD = 0.35  # cut when consecutive gap exceeds this
```

---

## Results So Far

### Threshold Search

| Run | Threshold | Faithfulness | Relevancy | Recall | Precision |
|---|---|---|---|---|---|
| Phase 2 Per-Card (baseline) | — | 0.574 | 0.733 | 0.869 | 0.663 |
| Score Gap 0.35 | 0.35 | **0.590** | **0.757** | 0.815 | **0.664** |
| Score Gap 0.45 | 0.45 | 0.595 | 0.765 | 0.815 | 0.656 |

**0.35 is the better threshold.** At 0.45 the cross_card category degraded significantly (precision collapsed to 0.000, faithfulness dropped from 0.533 → 0.375) because the looser cutoff retained borderline chunks that added noise for comparison queries.

### Per-Category Impact (Gap 0.35 vs Phase 2 Baseline)

| Category | Metric | Phase 2 Baseline | Phase 3 Gap 0.35 | Delta |
|---|---|---|---|---|
| cross_card | faithfulness | 0.400 | **0.533** | **+0.133** |
| cross_card | answer_relevancy | 0.670 | 0.686 | +0.016 |
| cross_card | context_recall | 0.708 | 0.583 | -0.125 |
| benefit_detail | faithfulness | 0.777 | 0.784 | +0.007 |
| exact_term | faithfulness | 0.917 | 0.917 | 0.000 |
| negative | answer_relevancy | 0.104 | 0.238 | +0.134 |
| simple_fact | faithfulness | 0.333 | 0.300 | -0.033 |

**Key finding:** The score gap cutoff improved cross_card faithfulness (+0.133) by reducing the noise chunks that caused the LLM to conflate information across cards. The recall hit (-0.125 on cross_card) is the precision-recall tradeoff: we're cutting some chunks that carried relevant information.

**Tried and failed:** Quote-forcing in the system prompt (instructing the LLM to copy amounts verbatim) backfired — faithfulness dropped overall (0.574 → 0.532). Root cause: verbatim copying produces awkward phrasing that RAGAS's claim decomposer can't trace back cleanly, particularly for exact_term (-0.250) and cross_card (-0.150).

---

## Root Causes Still Remaining

| Problem | Category | Evidence |
|---|---|---|
| cross_card recall hit | cross_card | Recall 0.583 with gap cutoff vs 0.708 baseline — some relevant chunks cut |
| cross_card precision still 0.083 | cross_card | Even with both cards retrieved, RAGAS can't attribute claims cleanly |
| Faithfulness plateau ~0.59 | all | Paraphrasing + RAGAS evaluator noise (temperature=0 ≠ deterministic) |
| Negative faithfulness stuck at 0.30 | negative | BM25 noise + RAGAS volatility on identical answers |

---

## What's Next

### 2. Reranker score-based abstention (negative questions)
If the top reranked chunk score falls below a threshold (~0.25), short-circuit to "this benefit is not listed" instead of synthesizing from low-confidence chunks. Directly targets negative faithfulness.

### 3. Experiment tracking with W&B
Add Weights & Biases logging to capture per-run metrics, per-question scores, and retrieved chunk traces in a queryable UI rather than raw CSV files.

### 4. Parent-child chunking (requires re-indexing)
512-token chunks frequently split a single benefit description. Parent-child retrieval fetches small chunks for precision but sends the full parent section to the LLM. Would improve recall on benefit_detail and cross_card.

### 5. Sub-question decomposition for cross-card queries
Decompose "compare X and Y on Z" into two independent sub-queries, answer each with 1-2 high-precision chunks, then synthesize. Directly fixes cross_card precision by ensuring each sub-answer is grounded in a single card's chunks.
