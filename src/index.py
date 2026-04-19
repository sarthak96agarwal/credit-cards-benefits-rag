"""
Phase 1: Naive indexing pipeline
- Loads PDFs from data/pdfs/
- Chunks with fixed 512-token size (intentionally naive — we'll fix this later)
- Embeds with OpenAI text-embedding-3-small
- Stores in ChromaDB
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path("data/pdfs")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "cc_benefits"

# Naive chunk settings — intentionally simple for Phase 1
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ── LlamaIndex global settings ────────────────────────────────────────────────

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def get_card_name_from_filename(filename: str) -> str:
    """
    Derive a human-readable card name from the PDF filename.
    Convention: name your PDFs like 'chase_sapphire_reserve.pdf'
    """
    stem = Path(filename).stem
    return stem.replace("_", " ").title()


def load_and_index():
    # Validate PDF directory
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"\n⚠️  No PDFs found in {PDF_DIR}/")
        print("   Add your credit card benefits PDFs there and re-run.")
        print("   Naming convention: chase_sapphire_reserve.pdf, amex_gold.pdf, etc.\n")
        return None

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"\n📄 Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"   - {f.name}")

    # Load documents — attach card_name metadata per file
    all_documents = []
    for pdf_path in pdf_files:
        card_name = get_card_name_from_filename(pdf_path.name)
        print(f"\n⏳ Loading: {card_name}...")

        docs = SimpleDirectoryReader(
            input_files=[str(pdf_path)],
            file_metadata=lambda path: {"card_name": card_name, "source": Path(path).name},
        ).load_data()

        print(f"   Loaded {len(docs)} page(s)")
        all_documents.extend(docs)

    print(f"\n✅ Total pages loaded: {len(all_documents)}")

    # Set up ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clear existing collection so re-runs start fresh
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index — this embeds all chunks and stores them
    print("\n⏳ Embedding and indexing chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print("\n✅ Indexing complete! ChromaDB stored at:", CHROMA_DIR)
    print("   Run `streamlit run src/app.py` to start the chatbot.\n")
    return index


if __name__ == "__main__":
    load_and_index()
