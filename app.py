# app.py
import argparse
from core.zero_shot_prompting import zero_shot_answer

def run_interactive():
    print("CryptoInsiteAI — Zero-shot prompting demo (offline). Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye 👋")
            break

        # Ask for optional output format (user can press Enter to skip)
        out_fmt = input("Optional output format (e.g. 'One-line summary' or 'JSON'): ").strip() or None
        constraints = input("Optional constraints (e.g. 'max 50 words'): ").strip() or None

        prompt_text, answer_text = zero_shot_answer(
            user_question=q,
            output_format=out_fmt,
            constraints=constraints,
            use_openai=False  # toggle true to attempt OpenAI call (needs env var + openai lib)
        )

        print("\n--- Built Prompt (zero-shot) ---")
        print(prompt_text)
        print("\n--- Model Answer ---")
        print(answer_text)

def run_example_once():
    q = "What is Bitcoin?"
    prompt_text, answer_text = zero_shot_answer(q)
    print(prompt_text)
    print("---")
    print(answer_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoInsiteAI zero-shot demo")
    parser.add_argument("--example", action="store_true", help="Run a single example and exit")
    args = parser.parse_args()
    if args.example:
        run_example_once()
    else:
        run_interactive()
