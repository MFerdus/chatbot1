"""Load the project PDFs and idempotently upsert them into Pinecone."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from src.helper import filter_to_minimal_docs, get_vector_store, load_pdf_file, text_split

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def document_id(document) -> str:
    identity = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> None:
    data_dir = ROOT_DIR / os.getenv("DATA_DIR", "data")
    print(f"Loading PDFs from {data_dir} ...")
    documents = filter_to_minimal_docs(load_pdf_file(data_dir))
    chunks = text_split(documents)
    vector_store = get_vector_store()

    batch_size = int(os.getenv("INDEX_BATCH_SIZE", "100"))
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_documents(batch, ids=[document_id(doc) for doc in batch])
        print(f"Indexed {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")

    print("Pinecone indexing completed successfully.")


if __name__ == "__main__":
    main()
