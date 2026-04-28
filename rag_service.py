"""
RAG microservice — thin FastAPI wrapper around the query pipeline.

Start:
    uvicorn rag_service:app --host 0.0.0.0 --port $PORT

Env vars required:
    OPENAI_API_KEY, COHERE_API_KEY (optional), PINECONE_API_KEY, PINECONE_INDEX_NAME
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
    return rag_query(req.question, _engine)
