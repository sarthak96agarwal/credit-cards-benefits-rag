"""
Phase 1: Naive query engine
- Loads the ChromaDB index built by index.py
- Retrieves top-5 chunks by cosine similarity (pure dense, no hybrid yet)
- Passes chunks + query to GPT-4o-mini
- Returns answer + source chunks for inspection
"""

from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "cc_benefits"
TOP_K = 5  # number of chunks to retrieve

# Naive system prompt — intentionally minimal for Phase 1
# You will notice hallucinations. We'll fix this in Phase 2.
SYSTEM_PROMPT = """You are a credit card benefits assistant.

Answer ONLY using the provided context from the benefits documents.

IMPORTANT RULES:
- If the context does not mention a benefit, say clearly: 
  "This benefit is not mentioned in the [Card Name] benefits guide."
- Never assume a benefit exists if it is not explicitly stated in the context.
- Never answer from general knowledge about credit cards.
- Always state which card you are referring to."""

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, system_prompt=SYSTEM_PROMPT)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def load_query_engine():
    """Load the persisted ChromaDB index and return a query engine."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_DIR}. Run `python src/index.py` first."
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)

    # SimilarityPostprocessor filters out chunks below a similarity threshold
    # Phase 1: threshold is low (0.3) — we'll raise this in Phase 2
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor],
    )

    return query_engine


def query(question: str, query_engine=None):
    """
    Ask a question and return the answer + source chunks.
    
    Returns:
        dict with keys:
            - answer (str)
            - sources (list of dicts with text, card_name, score, page)
    """
    if query_engine is None:
        query_engine = load_query_engine()

    response = query_engine.query(question)

    sources = []
    for node in response.source_nodes:
        sources.append({
            "text": node.node.text,
            "card_name": node.node.metadata.get("card_name", "Unknown"),
            "source_file": node.node.metadata.get("source", ""),
            "page": node.node.metadata.get("page_label", "?"),
            "score": round(node.score, 3) if node.score else None,
        })

    return {
        "answer": str(response),
        "sources": sources,
    }


if __name__ == "__main__":
    # Quick CLI test
    print("\n💳 Credit Card Benefits RAG — Phase 1\n")
    engine = load_query_engine()

    test_questions = [
        "What is the annual fee?",
        "Is there airport lounge access?",
        "Does the card cover rental car insurance?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = query(q, engine)
        print(f"A: {result['answer']}")
        print(f"   Sources: {[s['card_name'] for s in result['sources']]}")
        print(f"   Scores:  {[s['score'] for s in result['sources']]}")
        print()
