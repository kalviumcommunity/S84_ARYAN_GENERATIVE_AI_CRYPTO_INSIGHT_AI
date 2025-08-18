"""
core/chain_of_thought.py
========================
Implements Chain-of-Thought prompting for CryptoInsightAI.
This technique guides the AI to reason step-by-step
before producing a final answer.
"""

import prompting


def chain_of_thought_prompt(user_question: str, show_reasoning: bool = True):
    """
    Generates AI response using Chain-of-Thought style reasoning.
    
    :param user_question: str - User's query.
    :param show_reasoning: bool - Whether to display reasoning in the final answer.
    :return: str - AI response
    """
    if show_reasoning:
        # Explicitly tell AI to show reasoning steps
        system_prompt = (
            "You are CryptoInsightAI. Think through the problem step-by-step, "
            "explain your reasoning clearly, and then provide the final answer."
        )
    else:
        # Instruct AI to think internally (hidden reasoning)
        system_prompt = (
            "You are CryptoInsightAI. Think carefully step-by-step internally, "
            "but only output the final answer without showing your reasoning."
        )

    return prompting.system_user_prompt(system_prompt, user_question)


# Example usage
if __name__ == "__main__":
    question = "If I have 3 apples and eat one, how many are left?"

    print("--- CoT WITH reasoning ---")
    print(chain_of_thought_prompt(question, show_reasoning=True))

    print("\n--- CoT WITHOUT reasoning ---")
    print(chain_of_thought_prompt(question, show_reasoning=False))
