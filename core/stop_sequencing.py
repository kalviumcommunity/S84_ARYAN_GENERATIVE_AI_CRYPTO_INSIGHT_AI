"""
core/stop_sequence.py
=====================
Implements stop-sequence control using the OpenAI API.

Stop sequences allow us to *cut off* the model’s response when certain
keywords or markers appear (e.g., "END", "---", etc.).

Useful for:
- Structuring outputs (like crypto reports ending at a marker).
- Preventing unnecessary text generation.
- Ensuring clean integration in downstream workflows.
"""

import os
import openai
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)

# Configure OpenAI client
if USE_OPENAI:
    openai.api_key = OPENAI_API_KEY


# ------------------------
# Helpers: Dummy + Stop Apply
# ------------------------
def _truncate_at_stop_sequences(text: str, stop_sequences):
    if not stop_sequences:
        return text
    earliest = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1:
            earliest = min(earliest, idx)
    return text[:earliest]


def _dummy_stop_output(prompt: str, stop_sequences):
    base = (
        "[Dummy Stop-Sequence Output]\n"
        f"Prompt: {prompt}\n"
        "Sample: This is a deterministic sample that demonstrates how stop sequences cut off output. "
        "We will include markers like END and --- to showcase truncation. END --- Continued text that should not appear."
    )
    return _truncate_at_stop_sequences(base, stop_sequences)


# ------------------------
# Stop Sequence Generator
# ------------------------
def generate_with_stop_sequence(prompt, stop_sequences):
    """
    Generates text with stop sequences using OpenAI Chat Completions.

    :param prompt: The text prompt for the LLM.
    :param stop_sequences: List of strings where generation will stop.
    :return: Truncated response string.
    """
    if not USE_OPENAI:
        return _dummy_stop_output(prompt, stop_sequences)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            # OpenAI supports a `stop` list for stop sequences
            stop=stop_sequences,
            max_tokens=256,
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        return content
    except Exception:
        # Fallback ensures terminal output even on API failure
        return _dummy_stop_output(prompt, stop_sequences)


# ------------------------
# Demo Runs
# ------------------------
if __name__ == "__main__":
    # Example: Force model to stop at "END" or "---"
    prompt_text = (
        "Generate a short market insight with 2 bullets and then say END.\n"
        "Insight:"
    )

    print("\n--- STOP SEQUENCE DEMO ---")
    print(generate_with_stop_sequence(prompt_text, ["END", "---"]))
