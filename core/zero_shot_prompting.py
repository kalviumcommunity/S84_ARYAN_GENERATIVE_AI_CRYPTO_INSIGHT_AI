"""
Single canonical implementation of zero-shot prompting.
This file replaces the previous numbered module for simpler imports and execution.
"""

from typing import Tuple
import os
import sys
from dotenv import load_dotenv

# Ensure local imports work when executed as a script from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    # When imported as part of the package
    from .prompting import build_zero_shot_prompt, system_user_prompt  # type: ignore
except Exception:
    # When run as a standalone script
    from prompting import build_zero_shot_prompt, system_user_prompt  # type: ignore

# Load environment variables from a .env file if present (including OPENAI_API_KEY)
load_dotenv()

def mock_llm_generate(prompt_text: str) -> str:
    """
    A tiny rule-based 'mock LLM' for offline demo.
    Replace this with a real model call later.
    """
    q = prompt_text.lower()
    if "what is bitcoin" in q or "what is btc" in q:
        return ("Bitcoin (BTC) is a decentralized digital currency introduced in 2009. "
                "It runs on a proof-of-work blockchain and is used as digital store of value.")
    if "price of bitcoin" in q or "btc price" in q:
        return "Mocked price: BTC = $43,200 (demo). Replace with live API for real prices."
    if "what is ethereum" in q or "eth" in q:
        return ("Ethereum (ETH) is a programmable blockchain supporting smart contracts and dapps.")
    if "how to buy" in q and "bitcoin" in q:
        return ("Common steps: (1) choose an exchange, (2) create and verify account (KYC), "
                "(3) deposit funds, (4) place a buy order, (5) move to a secure wallet.")
    # fallback
    return "Sorry — I don't have a prepared answer for that in the offline demo."

def zero_shot_answer(user_question: str,
                     task_instruction: str = "Answer the user's question about cryptocurrency concisely.",
                     output_format: str = None,
                     constraints: str = None,
                     use_openai: bool = False) -> Tuple[str, str]:
    """
    Build zero-shot prompt and generate answer (mocked by default).
    Returns (prompt_text, answer_text).

    If use_openai=True, tries to call OpenAI's API (code included but optional).
    """
    prompt_text = build_zero_shot_prompt(task_instruction, user_question, output_format, constraints)

    if use_openai:
        try:
            # Prefer modern OpenAI client (openai>=1.0.0)
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment or .env.")

            client = OpenAI(api_key=api_key)
            preferred_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            def _create_completion(model_name: str):
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are CryptoInsiteAI — answer concisely."},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )

            try:
                resp = _create_completion(preferred_model)
            except Exception:
                # Retry with a safe fallback model name if the preferred one is unavailable
                fallback_model = "gpt-4o" if preferred_model != "gpt-4o" else preferred_model
                resp = _create_completion(fallback_model)

            text = resp.choices[0].message.content.strip()
            return prompt_text, text
        except Exception as e:
            # If OpenAI isn't available, fallback to mock and notify user in the answer.
            fallback = f"[OpenAI failed or not configured: {e}]\n\n" + mock_llm_generate(prompt_text)
            return prompt_text, fallback

    # default: use mock generator
    answer = mock_llm_generate(prompt_text)
    return prompt_text, answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run zero-shot answer (standalone)")
    parser.add_argument("-q", "--question", default="What is Bitcoin?", help="User question")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    prompt_text, answer_text = zero_shot_answer(
        user_question=args.question,
        use_openai=args.openai,
    )
    print("\n--- Built Prompt (zero-shot) ---")
    print(prompt_text)
    print("\n--- Model Answer ---")
    print(answer_text)
