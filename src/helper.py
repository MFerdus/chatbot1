"""Document loading, chunking, embeddings, and Pinecone helpers."""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Iterable
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_DIMENSIONS = 1536

def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to .env or the runtime environment.")
    return value

def load_pdf_file(data: str | Path) -> list[Document]:
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Data directory was not found: {path}")
    documents = DirectoryLoader(str(path), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    if not documents:
        raise ValueError(f"No PDF documents were found in {path}")
    return documents

def filter_to_minimal_docs(docs: Iterable[Document]) -> list[Document]:
    result = []
    for doc in docs:
        text = doc.page_content.strip()
        if text:
            result.append(Document(page_content=text, metadata={"source": str(doc.metadata.get("source", "unknown")), "page": int(doc.metadata.get("page", 0))}))
    return result

def text_split(extracted_data: Iterable[Document]) -> list[Document]:
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(list(extracted_data))
    if not chunks:
        raise ValueError("No text chunks were produced from the documents.")
    return chunks

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), api_key=require_env("OPENAI_API_KEY"))

def ensure_pinecone_index() -> tuple[Pinecone, str]:
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot").strip()
    client = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    indexes = client.list_indexes()
    names = getattr(indexes, "names", None)
    if names is not None:
        existing = set(names() if callable(names) else names)
    else:
        existing = {item["name"] for item in indexes}
    if index_name not in existing:
        client.create_index(name=index_name, dimension=EMBEDDING_DIMENSIONS, metric="cosine", spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1")))
        deadline = time.time() + 120
        while time.time() < deadline:
            status = client.describe_index(index_name).status
            ready = getattr(status, "ready", None)
            if ready is None and isinstance(status, dict):
                ready = status.get("ready", False)
            if ready:
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Pinecone index '{index_name}' did not become ready.")
    description = client.describe_index(index_name)
    if int(description.dimension) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(f"Index '{index_name}' has dimension {description.dimension}; expected {EMBEDDING_DIMENSIONS}. Use a new index name or recreate it.")
    return client, index_name

def get_vector_store() -> PineconeVectorStore:
    _, index_name = ensure_pinecone_index()
    return PineconeVectorStore(index_name=index_name, embedding=get_embeddings(), namespace=os.getenv("PINECONE_NAMESPACE", "medical-knowledge"))

download_hugging_face_embeddings = get_embeddings
