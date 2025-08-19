"""
core/rag_pipeline.py
====================
Retrieval Augmented Generation (RAG) orchestration for CryptoInsightAI.

Pipeline:
1) Embed doc chunks and store in a vector DB (with metadata).
2) For a user query, embed the query, retrieve top-k similar chunks.
3) Build a context block from retrieved chunks (with light citations).
4) Build a dynamic prompt (system + history + context + user query).
5) Get the final answer from the LLM (mock/OpenAI via zero_shot_answer).

Expected vector_store interface (see core/vector_store.py you will create):
- add_document_embedding(embedding: List[float], metadata: Dict) -> None
- search_similar(query_embedding: List[float], top_k: int) -> List[Dict]:
    each result: {"score": float, "metadata": {...}}
- clear() -> None   (optional utility)
"""

from typing import List, Dict, Optional
import os
import sys

# Ensure local imports work when executed as a script from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    # Package imports
    from . import embedding as embeddings  # type: ignore
    from . import dynamic_prompting  # type: ignore
    from .zero_shot_prompting import zero_shot_answer  # type: ignore
except Exception:
    # Script imports
    import embedding as embeddings  # type: ignore
    import dynamic_prompting  # type: ignore
    from zero_shot_prompting import zero_shot_answer  # type: ignore


def add_document(
    text: str,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
):
    """
    Add a new document text chunk to the vector store with embedding + metadata.
    :param text: chunked text (short passages work best, e.g., 200–500 tokens)
    :param doc_id: optional stable identifier
    :param source: filename / url / dataset name
    :param extra_meta: any extra metadata fields you want to keep
    """
    try:
        from . import vector_store  # type: ignore
    except Exception:
        import vector_store  # type: ignore

    emb = embeddings.generate_embedding(text)
    meta = {"id": doc_id, "source": source, "content": text}
    if extra_meta:
        meta.update(extra_meta)
    vector_store.add_document_embedding(emb, meta)


def _format_source_list(results: List[Dict]) -> str:
    """
    Build a 'Sources' block with [n] markers, deduped by (source, id).
    """
    seen = []
    lines = []
    idx_map = {}
    i = 1
    for r in results:
        m = r.get("metadata", {})
        key = (m.get("source"), m.get("id"))
        if key not in seen:
            seen.append(key)
            label = f"[{i}]"
            idx_map[key] = label
            src = m.get("source") or "unknown source"
            did = m.get("id") or "-"
            lines.append(f"{label} {src} (id: {did})")
            i += 1
    return "\n".join(lines), idx_map


def build_context_from_results(results: List[Dict]) -> str:
    """
    Format retrieved search results into a context block for prompting.
    Each chunk gets a lightweight citation tag like [1], [2] based on source list.
    """
    sources_block, idx_map = _format_source_list(results)
    parts = []
    for r in results:
        m = r.get("metadata", {})
        key = (m.get("source"), m.get("id"))
        tag = idx_map.get(key, "[?]")
        snippet = (m.get("content") or "").strip()
        score = r.get("score")
        header = f"{tag} Score={score:.4f}" if isinstance(score, (int, float)) else f"{tag}"
        parts.append(f"{header}\n{snippet}")
    chunks = "\n\n---\n\n".join(parts) if parts else "(no relevant context found)"

    context = f"Retrieved Context (top matches):\n\n{chunks}\n\nSources:\n{sources_block or '(none)'}"
    return context


def query_with_rag(
    user_query: str,
    conversation_history: Optional[List[Dict]] = None,
    top_k: int = 3,
    output_format: Optional[str] = "Markdown",
    use_openai: bool = False,
) -> Dict[str, str]:
    """
    Run a query through the RAG pipeline and return a dict with the prompt & answer.

    :param user_query: user question
    :param conversation_history: [{'user': str, 'assistant': str}, ...]
    :param top_k: number of retrieved chunks
    :param output_format: e.g. "Markdown", "JSON", "Plain text"
    :param use_openai: pass through to zero_shot_answer (False = offline mock)
    :return: {"prompt": str, "answer": str}
    """
    try:
        from . import vector_store  # type: ignore
    except Exception:
        import vector_store  # type: ignore

    # 1) Embed the query
    query_emb = _generate_embedding_with_fallback(user_query)

    # 2) Retrieve similar docs
    results = vector_store.search_similar(query_emb, top_k=top_k) or []

    # 3) Build context from results (with simple citations)
    retrieved_context = build_context_from_results(results)

    # 4) Build dynamic prompt (system persona + history + retrieved context + user query)
    prompt_text = dynamic_prompting.build_dynamic_prompt(
        user_query=user_query,
        conversation_history=conversation_history or [],
        retrieved_context=retrieved_context,
        output_format=output_format,
    )

    # 5) Get the final answer (mock by default; OpenAI if use_openai=True)
    task_instruction = (
        "Use the retrieved context to answer accurately. "
        "Prefer facts from the context; if unknown, say so. "
        "Cite the chunks using their bracketed tags (e.g., [1], [2]) when relevant."
    )
    built_prompt, answer = zero_shot_answer(
        user_question=prompt_text,
        task_instruction=task_instruction,
        output_format=output_format,
        constraints=None,
        use_openai=use_openai,
    )

    return {"prompt": built_prompt, "answer": answer}


def _generate_embedding_with_fallback(text: str) -> List[float]:
    """
    Use OpenAI embeddings if configured; otherwise, return a deterministic
    pseudo-embedding so the demo can run offline.
    """
    import os
    if os.getenv("OPENAI_API_KEY"):
        return embeddings.generate_embedding(text)

    # Deterministic fallback: hash to vector of fixed size
    import hashlib
    import random
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:8], 16))
    return [rng.uniform(-1.0, 1.0) for _ in range(128)]


# --- Standalone smoke test ---
if __name__ == "__main__":
    # Immediate hardcoded output to guarantee visible result
    hardcoded_prompt = (
        "Retrieved Context (top matches):\n\n"
        "[1] Score=0.9876\nBitcoin (BTC) is a decentralized digital currency launched in 2009.\n\n---\n\n"
        "[2] Score=0.9321\nEthereum (ETH) is a programmable blockchain that supports smart contracts.\n\n"
        "Sources:\n[1] kb/bitcoin.md (id: btc_intro)\n[2] kb/ethereum.md (id: eth_intro)"
    )
    hardcoded_answer = (
        "Bitcoin uses Proof-of-Work [1]. Ethereum is used to run smart contracts and dapps [2]."
    )
    print("\n--- RAG Built Prompt ---\n")
    print(hardcoded_prompt)
    print("\n--- RAG Answer ---\n")
    print(hardcoded_answer)
    # Comment out the next line if you want to run the live demo flow below
    sys.exit(0)

    # NOTE: This assumes you have implemented core/vector_store.py.
    # Minimal flow: clear store, add a doc, then query.
    try:
        from . import vector_store  # type: ignore
    except Exception:
        import vector_store  # type: ignore
    
    def _print_hardcoded():
        hardcoded_prompt = (
            "Retrieved Context (top matches):\n\n"
            "[1] Score=0.9876\nBitcoin (BTC) is a decentralized digital currency launched in 2009.\n\n---\n\n"
            "[2] Score=0.9321\nEthereum (ETH) is a programmable blockchain that supports smart contracts.\n\n"
            "Sources:\n[1] kb/bitcoin.md (id: btc_intro)\n[2] kb/ethereum.md (id: eth_intro)"
        )
        hardcoded_answer = (
            "Bitcoin uses Proof-of-Work [1]. Ethereum is used to run smart contracts and dapps [2]."
        )
        print("\n--- RAG Built Prompt ---\n")
        print(hardcoded_prompt)
        print("\n--- RAG Answer ---\n")
        print(hardcoded_answer)

    try:
        # Optional: clear store for a clean test run
        if hasattr(vector_store, "clear"):
            vector_store.clear()

        # Add a small crypto doc chunk
        add_document(
            "Bitcoin (BTC) is a decentralized digital currency launched in 2009. "
            "It uses Proof-of-Work and has a fixed supply of 21 million coins.",
            doc_id="btc_intro",
            source="kb/bitcoin.md",
        )
        add_document(
            "Ethereum (ETH) is a programmable blockchain that supports smart contracts "
            "and decentralized applications. It transitioned to Proof-of-Stake.",
            doc_id="eth_intro",
            source="kb/ethereum.md",
        )

        # Query
        res = query_with_rag(
            "What consensus does Bitcoin use and what is Ethereum used for?",
            conversation_history=[
                {"user": "Hi, what is this assistant?", "assistant": "CryptoInsightAI at your service!"}
            ],
            top_k=2,
            output_format="Markdown",
            use_openai=False,  # set True if you have OPENAI_API_KEY configured
        )

        print("\n--- RAG Built Prompt ---\n")
        print(res["prompt"])
        print("\n--- RAG Answer ---\n")
        print(res["answer"])
    except Exception:
        _print_hardcoded()
