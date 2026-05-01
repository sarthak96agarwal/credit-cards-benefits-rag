"""
RAG microservice — thin FastAPI wrapper around the query pipeline.

Start:
    uvicorn rag_service:app --host 0.0.0.0 --port $PORT

Env vars required:
    OPENAI_API_KEY, COHERE_API_KEY (optional), PINECONE_API_KEY, PINECONE_INDEX_NAME
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL (optional, enables tracing)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from query import load_query_engine, query as rag_query  # noqa: E402

# ── Langfuse client — gracefully disabled if keys not set ─────────────────────
try:
    from langfuse import Langfuse
    _langfuse = Langfuse()
except Exception:
    _langfuse = None

_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = load_query_engine()
    yield


app = FastAPI(title="CC Benefits RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """
    Returns:
        {
            "answer":         str,
            "sources":        list of {text, card_name, source_file, page, score},
            "detected_cards": list of card names,
        }
    """
    if _langfuse:
        # Use start_as_current_observation so the LlamaIndex instrumentor (retrieval,
        # LLM call, embeddings) automatically nests its spans under this root span.
        # Using langfuse.trace() + trace.update() does NOT set the active context,
        # causing LlamaIndex spans to float as separate root traces.
        with _langfuse.start_as_current_observation(
            as_type="span",
            name="rag-query",
            input={"question": req.question},
        ) as span:
            result = rag_query(req.question, _engine)
            span.update(
                output={"answer": result["answer"]},
                metadata={
                    "detected_cards": result["detected_cards"],
                    "num_sources": len(result["sources"]),
                    "retrieved_chunks": [
                        {
                            "card_name": s["card_name"],
                            "page": s["page"],
                            "score": s["score"],
                            "text": s["text"],
                        }
                        for s in result["sources"]
                    ],
                },
            )
        _langfuse.flush()
    else:
        result = rag_query(req.question, _engine)

    return result
