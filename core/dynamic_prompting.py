"""
core/dynamic_prompting.py
=========================
Dynamic prompt construction for CryptoInsightAI.
Builds prompts at runtime using memory (chat history), retrieved docs (like RAG),
and system instructions.
"""

import os
import sys

# Ensure local imports work when executed as a script from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    # When imported as part of the package
    from .zero_shot_prompting import zero_shot_answer  # type: ignore
except Exception:
    # When run as a standalone script
    from zero_shot_prompting import zero_shot_answer  # type: ignore

def build_dynamic_prompt(
    user_query: str,
    conversation_history=None,
    retrieved_context: str = None,
    output_format: str = None
) -> str:
    """
    Creates a dynamic prompt string from multiple components.
    :param user_query: str - Current user question or command.
    :param conversation_history: list of dicts [{'user': '', 'assistant': ''}, ...]
    :param retrieved_context: str - Extra context from APIs/RAG (market data, whitepapers).
    :param output_format: str - Desired output format (JSON, Markdown, etc.)
    :return: str - Final constructed prompt.
    """
    prompt_parts = []

    # 1. System persona
    system_instructions = (
        "You are CryptoInsightAI, an expert assistant in cryptocurrencies, "
        "blockchains, and digital assets. "
        "Provide concise, factually correct answers with helpful explanations. "
        "Cite credible sources or context when relevant."
    )
    prompt_parts.append(f"System: {system_instructions}")

    # 2. Conversation history
    if conversation_history:
        history_str = ""
        for turn in conversation_history:
            history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        prompt_parts.append(f"Conversation History:\n{history_str.strip()}")

    # 3. Retrieved context (from live APIs or RAG)
    if retrieved_context:
        prompt_parts.append(f"Additional Context:\n{retrieved_context}")

    # 4. Output formatting
    if output_format:
        prompt_parts.append(f"Please respond in {output_format} format.")

    # 5. Current user query
    prompt_parts.append(f"User: {user_query}")

    # Combine into one string
    final_prompt = "\n\n".join(prompt_parts)
    return final_prompt


# Simple test run
if __name__ == "__main__":
    history = [
        {"user": "What is Bitcoin?", "assistant": "Bitcoin is a decentralized digital currency."}
    ]
    context = "As of 2025, Bitcoin remains the most valuable cryptocurrency by market cap."
    query = "What is the current price of Bitcoin?"

    dynamic_prompt = build_dynamic_prompt(query, history, context, output_format="Markdown")

    print("\n--- DYNAMIC PROMPT ---")
    print(dynamic_prompt)

    # Send to mock/OpenAI via zero_shot_answer
    print("\n--- AI RESPONSE ---")
    _, response = zero_shot_answer(dynamic_prompt)
    print(response)
