# 💳 Credit Card Benefits RAG

A RAG chatbot that answers questions about your credit card benefits.
Built across phases to intentionally teach RAG concepts by breaking and fixing things.

---

## Phase 1: Naive RAG (start here)

### Setup

```bash
# 1. Clone / download this folder, then:
cd cc-benefits-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# Edit .env and paste your key: OPENAI_API_KEY=sk-...
```

### Add your PDFs

Download benefits guides from your card issuers and drop them in `data/pdfs/`.

Name them clearly — the filename becomes the card name in the UI:
```
data/pdfs/
  chase_sapphire_reserve.pdf
  amex_gold.pdf
  citi_double_cash.pdf
```

### Run

```bash
# Step 1: Index your PDFs (run once, or re-run when you add new PDFs)
python src/index.py

# Step 2: Launch the chatbot
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
cc-benefits-rag/
├── data/
│   ├── pdfs/            ← drop your benefits PDFs here
│   └── chroma_db/       ← auto-created by index.py
├── src/
│   ├── index.py         ← indexing pipeline (load → chunk → embed → store)
│   ├── query.py         ← query engine (retrieve → generate)
│   └── app.py           ← Streamlit UI
├── notebooks/           ← for experiments (add your own)
├── .env.example
├── requirements.txt
└── README.md
```

---

## What to observe in Phase 1 (intentional limitations)

Try these queries and note where it fails:

| Query | Expected failure |
|---|---|
| "Does my card cover rental cars?" | May hallucinate if chunk misses key detail |
| "What is the Priority Pass benefit?" | Exact term may not retrieve well |
| "Which card has better travel insurance?" | No cross-card comparison support |
| "What is the annual fee?" | May return chunks from wrong card |

Write down every failure — each one maps to a fix in Phase 2.

---

## Coming in Phase 2
- Sentence-aware chunking with section headers prepended
- Hybrid retrieval (BM25 + dense) with RRF merging
- Metadata filtering by card name
- Reranker (cross-encoder)
- Stricter hallucination guardrails
- RAGAS evaluation
