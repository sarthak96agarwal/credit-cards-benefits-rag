"""
Indexing pipeline — embeds PDFs and stores them in Pinecone.

Run once (or whenever PDFs change):
    python src/index.py

What it does:
  1. Loads PDFs from data/pdfs/, attaches card_name metadata per file
  2. Chunks with SentenceSplitter (512 tokens, 50 overlap)
  3. Embeds with OpenAI text-embedding-3-small
  4. Upserts vectors into Pinecone
  5. Saves raw node text to data/bm25_corpus.json for BM25 keyword search
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path("data/pdfs")
BM25_CORPUS_PATH = Path("data/bm25_corpus.json")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cc-benefits")

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
            file_metadata=lambda path, cn=card_name: {"card_name": cn, "source": Path(path).name},
        ).load_data()

        print(f"   Loaded {len(docs)} page(s)")
        all_documents.extend(docs)

    print(f"\n✅ Total pages loaded: {len(all_documents)}")

    # ── Connect to Pinecone ───────────────────────────────────────────────────
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Clear existing vectors so re-runs start fresh
    try:
        pinecone_index.delete(delete_all=True)
        print(f"\n🗑️  Cleared existing vectors in Pinecone index '{PINECONE_INDEX_NAME}'...")
    except Exception:
        print(f"\nℹ️  Index '{PINECONE_INDEX_NAME}' is empty — skipping clear.")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Parse documents into nodes ────────────────────────────────────────────
    # We parse first so we can (a) save the raw text for BM25 and (b) index
    # the same node objects to Pinecone.  VectorStoreIndex.from_documents()
    # doesn't populate the local docstore when using PineconeVectorStore, so
    # pulling nodes from index.docstore.docs after the fact always returns 0.
    print("\n⏳ Parsing documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    parsed_nodes = splitter.get_nodes_from_documents(all_documents, show_progress=True)
    print(f"   {len(parsed_nodes)} chunks created")

    # ── Save BM25 corpus ──────────────────────────────────────────────────────
    # Pinecone isn't designed for full-corpus scans, so we serialize raw nodes
    # to a JSON file for BM25 keyword retrieval.
    bm25_data = [
        {"id": node.node_id, "text": node.text, "metadata": node.metadata}
        for node in parsed_nodes
    ]
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "w") as f:
        json.dump(bm25_data, f, indent=2)
    print(f"   BM25 corpus saved → {BM25_CORPUS_PATH} ({len(bm25_data)} chunks)")

    # ── Build index ───────────────────────────────────────────────────────────
    print("\n⏳ Embedding and upserting to Pinecone...")
    index = VectorStoreIndex(
        parsed_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    nodes = bm25_data  # reuse for summary print below

    print(f"\n✅ Indexing complete!")
    print(f"   Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"   BM25 corpus: {BM25_CORPUS_PATH} ({len(nodes)} chunks)")
    print("\n   Run `python src/query.py` to test or `streamlit run src/app.py` for the UI.\n")
    return index


if __name__ == "__main__":
    load_and_index()
