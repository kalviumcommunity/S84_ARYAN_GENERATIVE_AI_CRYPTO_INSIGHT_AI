"""
core/embedding.py
=================
Embedding generation using OpenAI Embeddings for CryptoInsightAI.

This module generates numerical vector representations for a given text
using an OpenAI embedding model (default: `text-embedding-3-small`).

You can use these embeddings for:
- Semantic search
- Document similarity
- Retrieval Augmented Generation (RAG) pipeline
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def generate_embedding(text: str):
    """
    Generate an embedding vector for the input text using the OpenAI embedding model.
    Returns a list of floats.
    """
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
    except Exception as e:
        raise Exception(f"❌ Embedding API call failed: {e}") from e

    if not resp.data or not resp.data[0].embedding:
        raise Exception("❌ No embedding returned from OpenAI.")

    return resp.data[0].embedding


if __name__ == "__main__":
    # Standalone test
    sample_text = "What is the capital of France?"
    embedding = generate_embedding(sample_text)
    print(f"✅ Embedding vector length: {len(embedding)}")
    print("🔹 First 10 values:", embedding[:10])
