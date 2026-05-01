# Credit Card Benefits RAG

A production RAG pipeline that answers questions about credit card benefits — coverage limits, lounge access, trip insurance, and more. Built across phases to teach real production MLE concepts by incrementally improving a naive baseline.

---

## Monitoring & Services

| Service | Purpose | Link |
|---|---|---|
| **Pinecone** | Vector database (408 chunks, 1536-dim) | [console.pinecone.io](https://console.pinecone.io) |
| **Langfuse** | Trace explorer, online eval, scores dashboard | [cloud.langfuse.com](https://cloud.langfuse.com) |
| **Cohere** | Reranker API usage & rate limits | [dashboard.cohere.com](https://dashboard.cohere.com) |
| **OpenAI** | LLM + embedding usage & costs | [platform.openai.com/usage](https://platform.openai.com/usage) |
| **W&B** | Offline RAGAS eval experiment tracking | [wandb.ai/sarthak96agarwal-bloomberg/cc-benefits-rag](https://wandb.ai/sarthak96agarwal-bloomberg/cc-benefits-rag) |
| **Railway** | RAG microservice deployment | [railway.app](https://railway.app) |

---

## Pipeline Architecture

```
User question
     │
     ▼
Card Detection          ← fuzzy match against known card names
     │
     ▼
Hybrid Retrieval        ← per detected card (or unfiltered if none)
  ├── Vector Search     ← Pinecone ANN, top-8
  ├── BM25 Search       ← local JSON corpus, top-8
  ├── RRF Fusion        ← Reciprocal Rank Fusion (k=60)
  └── Cohere Rerank     ← cross-encoder, adaptive score-gap cutoff
     │
     ▼
Multi-card Interleave   ← if 2+ cards: [A1, B1, A2, B2, ...] (not sort-merge)
     │
     ▼
LLM Synthesis           ← gpt-4o-mini, strict grounding prompt
     │
     ▼
Response + Sources      ← answer, retrieved chunks, detected cards
     │
     ▼
Langfuse Trace          ← answer, retrieved_chunks, scores logged async
```

---

## Project Structure

```
cc-benefits-rag/
├── data/
│   ├── pdfs/               ← card benefits PDFs (gitignored)
│   └── bm25_corpus.json    ← serialized chunks for BM25 keyword search
├── src/
│   ├── config.py           ← all tunable constants (TOP_K, thresholds, models)
│   ├── index.py            ← indexing pipeline: PDF → chunks → Pinecone + BM25 JSON
│   ├── store.py            ← Pinecone index loader + BM25 corpus loader
│   ├── retriever.py        ← hybrid retrieval: vector + BM25 + RRF + rerank
│   ├── card_detector.py    ← fuzzy card name detection from question
│   ├── query.py            ← orchestration: detect → retrieve → synthesize
│   └── app.py              ← Streamlit UI (local testing)
├── eval/
│   ├── evaluate.py         ← offline RAGAS evaluation, logs to W&B
│   ├── test_dataset.py     ← ground-truth Q&A pairs
│   └── results/            ← per-run CSV outputs
├── rag_service.py          ← FastAPI microservice (production entry point)
├── railway.json            ← Railway start command config
└── requirements.txt
```

---

## Key Config (`src/config.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-small` | 1536-dim OpenAI embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Answer synthesis |
| `RERANK_MODEL` | `rerank-v3.5` | Cohere cross-encoder |
| `TOP_K` | 8 | Candidates per retriever per card |
| `RERANK_TOP_K` | 5 | Max chunks after reranking |
| `RERANK_MIN_K` | 2 | Floor — always keep at least 2 chunks |
| `SCORE_GAP_THRESHOLD` | 0.35 | Drop chunks when reranker gap exceeds this |
| `RRF_K` | 60 | RRF smoothing constant |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

Required env vars:
```
OPENAI_API_KEY=
COHERE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=cc-benefits
LANGFUSE_PUBLIC_KEY=       # optional — enables production tracing
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
WANDB_PROJECT=cc-benefits-rag
WANDB_ENTITY=
```

---

## Indexing (run once, or when PDFs change)

```bash
python src/index.py
```

- Clears existing Pinecone vectors and re-upserts
- Saves `data/bm25_corpus.json` (408 chunks across 6 cards)

---

## Running locally

```bash
# Query CLI
python src/query.py

# Streamlit UI
streamlit run src/app.py

# FastAPI service
uvicorn rag_service:app --port 8001
```

---

## Offline Evaluation

```bash
python eval/evaluate.py                      # run eval, save CSV
python eval/evaluate.py --upload-history     # upload past CSVs to W&B
```

Metrics: faithfulness, answer relevancy, context precision, context recall (RAGAS).
Results logged to W&B with per-question retrieved chunks and reranker scores.

---

## Production Deployment (Railway)

Start command (set in `railway.json`):
```
uvicorn rag_service:app --host 0.0.0.0 --port $PORT
```

API:
- `GET /health` — liveness check
- `POST /query` — `{"question": "..."}` → `{"answer", "sources", "detected_cards"}`

---

## Phase History

| Phase | What changed | Key metric |
|---|---|---|
| Phase 1 | Naive RAG — dense vector only, no filtering | Baseline |
| Phase 2 | Hybrid (BM25 + RRF) + Cohere reranker + card metadata filter + RAGAS eval | Faithfulness ↑ |
| Phase 3 | Adaptive score-gap cutoff + per-card interleaving for cross-card queries + W&B tracking | Cross-card faithfulness 0.175 → 0.533 |
| Phase 4 | Pinecone migration + FastAPI microservice + Langfuse online tracing | Production-ready |
