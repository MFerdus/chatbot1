"""VS Code-ready Flask application for the medical RAG chatbot."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.helper import get_vector_store
from src.prompt import SYSTEM_PROMPT


ROOT_DIR = Path(__file__).resolve().parent

# override=True prevents an old PowerShell environment variable from taking
# precedence over the corrected values in the project's .env file.
load_dotenv(ROOT_DIR / ".env", override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


def get_required_setting(name: str) -> str:
    """Return a clean required setting or raise a helpful configuration error."""
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"{name} is missing. Add it to {ROOT_DIR / '.env'}."
        )

    if '"' in value or "'" in value or "=" in value:
        raise RuntimeError(
            f"{name} is malformed. In .env, use {name}=your_actual_key "
            "with no quotation marks, spaces, or repeated variable name."
        )

    if name == "OPENAI_API_KEY" and not value.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY has an invalid prefix. Copy a current API key "
            "from the OpenAI API platform."
        )

    return value


def configuration_status() -> tuple[bool, str]:
    """Check configuration without contacting OpenAI or Pinecone."""
    try:
        get_required_setting("OPENAI_API_KEY")
        get_required_setting("PINECONE_API_KEY")
        return True, "configured"
    except RuntimeError as exc:
        return False, str(exc)


def format_documents(documents: list[Document]) -> str:
    """Convert retrieved documents into the context supplied to the LLM."""
    if not documents:
        return "No relevant context was retrieved."

    sections = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "unknown")
        sections.append(
            f"Source: {source}; page: {page}\n{document.page_content}"
        )
    return "\n\n".join(sections)


@lru_cache(maxsize=1)
def get_rag_chain() -> Any:
    """Create external clients only when the first question is submitted."""
    openai_api_key = get_required_setting("OPENAI_API_KEY")
    get_required_setting("PINECONE_API_KEY")

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("RETRIEVAL_K", "4"))},
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        api_key=openai_api_key,
        temperature=0,
        timeout=60,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    # LangChain 1.x LCEL pipeline. This replaces the removed legacy
    # retrieval-chain helper API.
    return (
        {
            "context": retriever | format_documents,
            "input": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


@app.get("/")
def index():
    """Render the chat interface."""
    return render_template("chat.html")


@app.get("/health")
def health():
    """Report startup configuration without exposing secrets."""
    configured, detail = configuration_status()
    return (
        jsonify(
            status="ok" if configured else "configuration_required",
            credentials_configured=configured,
            detail=detail,
            index=os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-hf384"),
            namespace=os.getenv("PINECONE_NAMESPACE", "medical-knowledge"),
        ),
        200 if configured else 503,
    )


@app.post("/get")
def chat():
    """Answer one form-encoded chat question."""
    message = request.form.get("msg", "").strip()

    if not message:
        return jsonify(error="Please enter a question."), 400

    if len(message) > 4000:
        return jsonify(
            error="The question is too long (maximum 4,000 characters)."
        ), 400

    try:
        answer = get_rag_chain().invoke(message)
        return jsonify(answer=answer or "No answer was generated.")
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
        return jsonify(error=str(exc)), 503
    except Exception:
        logger.exception("Chat request failed")
        return jsonify(
            error=(
                "The assistant is temporarily unavailable. Verify that "
                "store_index.py completed and check the server terminal."
            )
        ), 500


if __name__ == "__main__":
    configured, detail = configuration_status()
    if not configured:
        logger.warning("Configuration incomplete: %s", detail)

    app.run(
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8080")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
