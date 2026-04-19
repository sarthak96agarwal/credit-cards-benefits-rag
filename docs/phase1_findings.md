# Phase 1 - Naive RAG: Findings & Analysis

## Approach

### Architecture
- **Indexing**: PDFs loaded via `SimpleDirectoryReader`, chunked with `SentenceSplitter` (512 tokens, 50 token overlap), embedded with `text-embedding-3-small`, stored in ChromaDB
- **Retrieval**: Pure dense vector search (cosine similarity), top-5 chunks, similarity cutoff of 0.3
- **Generation**: `gpt-4o-mini` with temperature=0
- **No hybrid search, no reranking, no metadata filtering**

### System Prompt
```
You are a credit card benefits assistant.
Answer ONLY using the provided context from the benefits documents.

IMPORTANT RULES:
- If the context does not mention a benefit, say clearly:
  "This benefit is not mentioned in the [Card Name] benefits guide."
- Never assume a benefit exists if it is not explicitly stated in the context.
- Never answer from general knowledge about credit cards.
- Always state which card you are referring to.
```

### Evaluation Setup
- **28 test questions** across 6 cards (Amex Gold, Amex Platinum, Delta SkyMiles Gold, Bilt Palladium, Capital One Venture X, United Explorer)
- **5 categories**: simple_fact (5), exact_term (3), benefit_detail (11), negative (5), cross_card (4)
- **4 RAGAS metrics**: faithfulness, answer_relevancy, context_recall, context_precision
- **Evaluator LLM**: gpt-4o-mini via RAGAS 0.4.3

---

## Results

### Overall Scores
| Metric | Score |
|---|---|
| Faithfulness | 0.490 |
| Answer Relevancy | 0.653 |
| Context Recall | 0.796 |
| Context Precision | 0.556 |

### Per-Category Breakdown
| Category | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|
| benefit_detail | 0.742 | 0.904 | 0.955 | 0.776 |
| cross_card | 0.208 | 0.213 | 0.333 | 0.050 |
| exact_term | 1.000 | 0.764 | 1.000 | 0.733 |
| negative | 0.300 | 0.238 | 0.600 | 0.307 |
| simple_fact | 0.233 | 0.991 | 1.000 | 0.733 |

---

## Key Findings

### 1. Cross-card comparisons are the weakest area (all metrics near zero)
- **Context Precision: 0.050** - The retriever pulls chunks from one card but not both. Almost all retrieved chunks are irrelevant noise.
- **Context Recall: 0.333** - Only finding info for one of the two cards being compared.
- **Root cause**: A single vector search cannot reliably pull from two separate documents simultaneously. The query "how does X differ between Card A and Card B" retrieves chunks semantically similar to the overall topic, but not necessarily from both cards.

### 2. Low faithfulness on simple_fact (0.233) despite perfect context_recall (1.000)
- The retriever finds the right chunks (recall = 1.0), but they're not ranked highly (precision = 0.733).
- The LLM's answer is factually correct, but the specific claim (e.g., "$60,000") sometimes appears in a chunk that was retrieved but not the one RAGAS checks against — or the LLM recalls the number from general knowledge rather than the provided context.
- RAGAS marks this as unfaithful because it can't trace the claim back to the retrieved chunks.

### 3. Negative questions expose hallucination tendencies (faithfulness: 0.300)
- When asked "does Card X have benefit Y?" where the answer is no, two failure modes:
  - The retriever pulls chunks from a *different card* that does have the benefit, confusing the LLM.
  - The LLM correctly says "not mentioned" but RAGAS scores it low because the context *does* contain related (but wrong-card) information.

### 4. Benefit detail performs best (faithfulness: 0.742, recall: 0.955)
- Single-card, detail-oriented questions work well because the retriever can find the relevant section and the LLM has enough context to answer accurately.

### 5. Exact term matching works well for recall (1.000) but not precision (0.733)
- The retriever finds the right info but also returns irrelevant chunks alongside it, diluting the context.

---

## Root Causes

| Problem | Cause | Affected Categories |
|---|---|---|
| Wrong-card chunks retrieved | No metadata filtering; retriever treats all cards as one pool | cross_card, negative |
| Relevant chunks buried in noise | No reranking; top-5 by similarity alone | simple_fact, exact_term |
| Can't retrieve from two cards at once | Single vector query can't span documents | cross_card |
| LLM adds info not in context | Chunks are too small (512 tokens) and miss surrounding details | simple_fact, benefit_detail |

---

## What Phase 2 Should Address

1. **Metadata filtering** - Filter by card_name so the retriever only searches relevant documents. For cross-card queries, run separate retrievals per card.
2. **Hybrid search (BM25 + dense)** - Add keyword search to catch exact terms like "$60,000" or "Priority Pass" that pure vector search may rank poorly.
3. **Reranking** - Add a reranker to push relevant chunks to the top and improve context precision.
4. **Larger chunk sizes or parent-child chunking** - 512 tokens often splits a benefit description mid-sentence. Consider 1024 tokens or hierarchical chunking.
5. **Raise similarity cutoff** - Current 0.3 is too permissive, letting irrelevant chunks through.
