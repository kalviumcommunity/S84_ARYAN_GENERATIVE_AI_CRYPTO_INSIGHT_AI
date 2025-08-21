"""
core/sampling.py
================
CryptoInsightAI - Sampling & Generation Control

This module provides advanced text generation controls for AI-driven
crypto insights & analysis. It leverages the OpenAI API with:
- Temperature: Controls creativity/randomness.
- Top-K: Restricts output to top K probable tokens.
- Top-P: Nucleus sampling, considers tokens with cumulative prob >= p.

Use-cases in CryptoInsightAI:
- Generating market summaries with varied creativity.
- Exploring multiple perspectives on crypto trends.
- Producing diverse investment strategies or risk assessments.
"""

import os
import openai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


# ------------------------
# Common OpenAI API Caller
# ------------------------
def call_openai_with_config(prompt, generation_config):
    """
    Calls the OpenAI API with a given prompt and specified generation config.
    """
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY missing. Add it to your .env file.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            **generation_config
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise Exception(f"❌ OpenAI API call failed: {str(e)}")


# ------------------------
# Temperature Control
# ------------------------
def generate_with_temperature(prompt, temperature=1.0):
    """
    Controls randomness/creativity of responses.
    - Lower temp (0.2): factual, focused analysis.
    - Higher temp (0.8): speculative, creative insights.
    """
    return call_openai_with_config(
        prompt,
        generation_config={"temperature": temperature}
    )


# ------------------------
# Top-K Sampling
# ------------------------
def generate_with_top_k(prompt, top_k=40):
    """
    Limits output to top_k most likely tokens.
    - Small K: concise, safer summaries.
    - Large K: more variety in crypto insights.
    """
    return call_openai_with_config(
        prompt,
        generation_config={"top_k": top_k}
    )


# ------------------------
# Top-P (Nucleus) Sampling
# ------------------------
def generate_with_top_p(prompt, top_p=0.9):
    """
    Nucleus sampling — considers tokens until cumulative prob >= top_p.
    - Low top_p (0.3): deterministic, precise forecasts.
    - High top_p (0.9): speculative, diverse perspectives.
    """
    return call_openai_with_config(
        prompt,
        generation_config={"top_p": top_p}
    )


# ------------------------
# Demo Runs for CryptoInsightAI
# ------------------------
if __name__ == "__main__":
    test_prompt_temp = "Give a short market summary for Bitcoin and Ethereum."
    print("\n--- TEMPERATURE CONTROL ---")
    print("Temperature 0.2 (deterministic):\n", generate_with_temperature(test_prompt_temp, 0.2))
    print("\nTemperature 0.8 (creative):\n", generate_with_temperature(test_prompt_temp, 0.8))

    test_prompt_topk = "Explain the potential impact of blockchain on traditional banking."
    print("\n--- TOP-K CONTROL ---")
    print("Top-K 10 (focused):\n", generate_with_top_k(test_prompt_topk, 10))
    print("\nTop-K 80 (diverse):\n", generate_with_top_k(test_prompt_topk, 80))

    test_prompt_topp = "Predict the role of AI in cryptocurrency trading over the next 5 years."
    print("\n--- TOP-P CONTROL ---")
    print("Top-P 0.3 (precise):\n", generate_with_top_p(test_prompt_topp, 0.3))
    print("\nTop-P 0.9 (speculative):\n", generate_with_top_p(test_prompt_topp, 0.9))

   