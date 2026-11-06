from functools import lru_cache
from typing import List, Dict, Any

import ollama
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
OLLAMA_INDEX_NAME = "medical-agent-ollama"


# --------------------------------------------------
# PINECONE CONNECTION
# --------------------------------------------------

@lru_cache()
def get_pinecone():
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache()
def get_index():
    """Create/validate the index once, then reuse it."""
    pc = get_pinecone()

    if not pc.has_index(OLLAMA_INDEX_NAME):
        print(f"Creating Pinecone index: {OLLAMA_INDEX_NAME}")
        pc.create_index(
            name=OLLAMA_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    description = pc.describe_index(OLLAMA_INDEX_NAME)

    if description.dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Pinecone index dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, got {description.dimension}."
        )

    return pc.Index(OLLAMA_INDEX_NAME)


# Backwards-compatible name used by ingest.py
ensure_index = get_index


# --------------------------------------------------
# OLLAMA EMBEDDINGS
# --------------------------------------------------

def embed_documents(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[f"search_document: {text}" for text in texts],
    )

    embeddings = response.embeddings

    if len(embeddings) != len(texts):
        raise ValueError("Embedding count does not match document count.")

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"received {len(embedding)}."
            )

    return [list(e) for e in embeddings]


def embed_query(query: str) -> List[float]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=f"search_query: {query}",
    )

    embedding = response.embeddings[0]

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"received {len(embedding)}."
        )

    return list(embedding)


# --------------------------------------------------
# KNOWLEDGE BASE SEARCH
# --------------------------------------------------

def search_kb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    index = get_index()
    query_vector = embed_query(query)

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = []

    for match in result.matches:
        metadata = match.metadata or {}

        matches.append({
            "id": match.id,
            "score": match.score,
            "text": metadata.get("text", ""),
            "page": metadata.get("page", 0),
            "source": metadata.get("file", ""),
        })

    return matches