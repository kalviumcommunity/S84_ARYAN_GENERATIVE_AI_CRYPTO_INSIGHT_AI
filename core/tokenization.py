"""
core/tokens_and_tokenization.py
===============================
Demonstration of tokens and tokenization for CryptoInsightAI.
Uses Hugging Face's GPT-2 tokenizer to show how text is split into tokens
and how token counts are computed.
"""

from transformers import AutoTokenizer

# Load GPT-2 tokenizer (commonly used for demonstration of subword tokenization)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def get_tokens(text: str):
    """
    Return list of tokens for the input text.
    :param text: str - Input sentence/paragraph
    :return: list of tokens
    """
    return tokenizer.tokenize(text)


def count_tokens(text: str) -> int:
    """
    Return number of tokens for the input text.
    :param text: str - Input sentence/paragraph
    :return: int - token count
    """
    return len(tokenizer.encode(text))


if __name__ == "__main__":
    sample_text = "Python is a programming language and HTML is a markup language."
    print("📌 Text:", sample_text)
    print("🔹 Tokens:", get_tokens(sample_text))
    print("🔢 Number of tokens:", count_tokens(sample_text))
