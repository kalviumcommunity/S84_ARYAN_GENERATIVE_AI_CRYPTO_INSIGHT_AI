# core/prompting.py
from typing import Optional, Dict

def system_user_prompt(system_prompt: str, user_question: str) -> Dict[str, str]:
    """
    Minimal wrapper returning prompt components as a dict.
    """
    return {"system": system_prompt, "user": user_question}


def build_zero_shot_prompt(task_instruction: str,
                          user_question: str,
                          output_format: Optional[str] = None,
                          constraints: Optional[str] = None) -> str:
    """
    Build a single zero-shot prompt string combining system role, instruction,
    user question, optional output format and constraints.

    Zero-shot = no examples / no few-shot context.
    """
    pieces = []
    # System / role
    pieces.append(f"SYSTEM: You are CryptoInsiteAI — a concise, accurate crypto assistant.\n"
                  f"Task instruction: {task_instruction}")

    # Optional constraints
    if constraints:
        pieces.append(f"CONSTRAINTS: {constraints}")

    # Optional output format instructions
    if output_format:
        pieces.append(f"OUTPUT FORMAT: {output_format}")

    # The user question
    pieces.append(f"USER QUESTION: {user_question}")

    # Combine
    prompt = "\n\n".join(pieces)
    return prompt
