"""
ChromaDB store access.
Responsible for loading the vector index and fetching raw nodes for BM25.
"""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_DIR, COLLECTION_NAME


def load_index() -> tuple[VectorStoreIndex, chromadb.Collection]:
    """
    Load the persisted ChromaDB index.

    Returns:
        (index, collection) — index for vector search, collection for BM25 node fetch.

    Raises:
        FileNotFoundError if the index hasn't been built yet.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_DIR}. Run `python src/index.py` first."
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    return index, collection


def get_all_nodes(collection: chromadb.Collection) -> list[TextNode]:
    """
    Fetch every document node from ChromaDB as TextNode objects.

    Used to build the BM25 corpus. When the index is loaded via
    VectorStoreIndex.from_vector_store(), the in-memory docstore is empty —
    nodes live in ChromaDB — so we fetch them directly.
    """
    result = collection.get(include=["documents", "metadatas"])
    return [
        TextNode(id_=doc_id, text=text, metadata=metadata or {})
        for doc_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]
