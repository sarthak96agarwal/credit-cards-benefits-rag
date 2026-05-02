"""
RAG microservice — thin FastAPI wrapper around the query pipeline.

Start:
    uvicorn rag_service:app --host 0.0.0.0 --port $PORT

Env vars required:
    OPENAI_API_KEY, COHERE_API_KEY (optional), PINECONE_API_KEY, PINECONE_INDEX_NAME
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL (optional, enables tracing)
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import structlog.contextvars
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from query import load_query_engine, query as rag_query  # noqa: E402


# ── Logging ───────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    service  = os.getenv("SERVICE_NAME", "rag")
    is_tty   = sys.stdout.isatty()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso" if not is_tty else "%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
        lambda _, __, ed: {**ed, "service": service},
    ]

    renderer = structlog.dev.ConsoleRenderer() if is_tty else structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


_configure_logging()
log = structlog.get_logger("rag")


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
    log.info("startup", detail="Loading query engine...")
    _engine = load_query_engine()
    log.info("startup", detail="Query engine ready.")
    yield
    # Flush all buffered traces before the process exits
    if _langfuse:
        _langfuse.shutdown()


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
def query(req: QueryRequest, request: Request):
    """
    Returns:
        {
            "answer":         str,
            "sources":        list of {text, card_name, source_file, page, score},
            "detected_cards": list of card names,
        }
    """
    t0 = time.perf_counter()
    log.info("request_start", question=req.question)

    # If the agent passed a Langfuse trace context, attach this span to that trace
    trace_id    = request.headers.get("X-Langfuse-Trace-Id")
    parent_obs_id = request.headers.get("X-Langfuse-Parent-Obs-Id")
    trace_context = (
        {"trace_id": trace_id, "parent_span_id": parent_obs_id}
        if trace_id else None
    )

    def _run():
        return rag_query(req.question, _engine)

    try:
        if _langfuse:
            with _langfuse.start_as_current_observation(
                as_type="span",
                name="rag-query",
                input={"question": req.question},
                trace_context=trace_context,
            ) as span:
                structlog.contextvars.bind_contextvars(trace_id=span.trace_id)
                result = _run()
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
        else:
            result = _run()

        scores = [s["score"] for s in result["sources"] if s["score"] is not None]
        log.info(
            "request_end",
            question=req.question,
            detected_cards=result["detected_cards"],
            num_sources=len(result["sources"]),
            reranker_scores=scores,
            duration_s=round(time.perf_counter() - t0, 2),
        )
        return result

    except Exception as exc:
        log.error(
            "request_error",
            question=req.question,
            error=str(exc),
            duration_s=round(time.perf_counter() - t0, 2),
        )
        raise
