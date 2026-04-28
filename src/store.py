"""
Pinecone store access.
Responsible for loading the vector index and fetching raw nodes for BM25.

Vector search  → Pinecone (managed hosted vector DB)
BM25 corpus    → data/bm25_corpus.json (serialized at index time, loaded into memory)
"""

import json
import os

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import PINECONE_INDEX_NAME, BM25_CORPUS_PATH


def load_index() -> VectorStoreIndex:
    """
    Connect to the Pinecone index and return a VectorStoreIndex for similarity search.

    Raises:
        ValueError if PINECONE_API_KEY is not set.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set.")

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def get_all_nodes() -> list[TextNode]:
    """
    Load all document nodes from the BM25 corpus JSON file.

    The corpus is serialized during indexing (src/index.py) and committed to
    the repo. Pinecone is not designed for full-corpus scans, so BM25 keyword
    search uses this lightweight local file instead.

    Raises:
        FileNotFoundError if the corpus hasn't been built yet (run src/index.py).
    """
    if not BM25_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {BM25_CORPUS_PATH}. Run `python src/index.py` first."
        )

    with open(BM25_CORPUS_PATH) as f:
        data = json.load(f)

    return [
        TextNode(id_=n["id"], text=n["text"], metadata=n["metadata"])
        for n in data
    ]
