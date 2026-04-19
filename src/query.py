"""
Phase 2: Query engine with metadata filtering
- Detects which card(s) a query is about (via card_detector.py)
- Filters retrieval to only relevant card(s)
- For cross-card queries, retrieves per card and merges results
"""

from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from card_detector import detect_cards

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "cc_benefits"
TOP_K = 5  # number of chunks to retrieve per card

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


def _load_index():
    """Load the persisted ChromaDB index."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_DIR}. Run `python src/index.py` first."
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


def _build_retriever(index, card_name: str | None = None):
    """Build a retriever, optionally filtered to a specific card."""
    filters = None
    if card_name:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="card_name", value=card_name, operator=FilterOperator.EQ),
        ])

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
        filters=filters,
    )


def _retrieve_nodes(index, question: str, card_name: str | None = None):
    """Retrieve and post-process nodes for a single card (or all cards)."""
    retriever = _build_retriever(index, card_name)
    nodes = retriever.retrieve(question)

    # Filter out low-similarity chunks
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
    nodes = postprocessor.postprocess_nodes(nodes, query_str=question)

    return nodes


def load_query_engine():
    """Load the index — returned object is the index itself now."""
    return _load_index()


def query(question: str, query_engine=None):
    """
    Ask a question and return the answer + source chunks.

    Detects card names in the question:
    - Single card: filters retrieval to that card only
    - Multiple cards: retrieves per card separately, merges results
    - No card detected: retrieves from all cards (unfiltered)

    Returns:
        dict with keys:
            - answer (str)
            - sources (list of dicts with text, card_name, score, page)
            - detected_cards (list of card names detected in the question)
    """
    index = query_engine if query_engine is not None else _load_index()
    detected = detect_cards(question)

    if len(detected) == 0:
        # No card detected — unfiltered retrieval
        nodes = _retrieve_nodes(index, question)
    elif len(detected) == 1:
        # Single card — filter to that card
        nodes = _retrieve_nodes(index, question, card_name=detected[0])
    else:
        # Multiple cards — retrieve per card, merge, and sort by score
        all_nodes = []
        for card in detected:
            card_nodes = _retrieve_nodes(index, question, card_name=card)
            all_nodes.extend(card_nodes)
        # Sort by score descending, keep top results
        all_nodes.sort(key=lambda n: n.score or 0, reverse=True)
        nodes = all_nodes[:TOP_K * len(detected)]

    # Build the query engine with the pre-retrieved nodes
    from llama_index.core.response_synthesizers import get_response_synthesizer
    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(question, nodes=nodes)

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
        "detected_cards": detected,
    }


if __name__ == "__main__":
    # Quick CLI test
    print("\n💳 Credit Card Benefits RAG — Phase 2 (Metadata Filtering)\n")
    index = load_query_engine()

    test_questions = [
        "What is the auto rental coverage on the United Explorer Card?",
        "Does the Amex Gold card include airport lounge access?",
        "How does the Trip Delay differ between the Capital One Venture X and the United Explorer?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = query(q, index)
        print(f"A: {result['answer']}")
        print(f"   Detected cards: {result['detected_cards']}")
        print(f"   Sources: {[s['card_name'] for s in result['sources']]}")
        print(f"   Scores:  {[s['score'] for s in result['sources']]}")
        print()
