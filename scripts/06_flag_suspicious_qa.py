import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
INPUT_PATH = Path("data/synthetic_qa.jsonl")

HEDGE_WORDS = [
    "presumably", "probably", "likely", "i believe", "it seems",
    "might be", "possibly", "i think", "appears to",
]
MARKUP_PATTERNS = [r":attr:`", r":func:`", r":class:`", r"::\s*$", r"`.*`"]


def flag_entry(entry: dict) -> list[str]:
    """Return reasons this entry looks suspicious, or [] if clean."""
    reasons = []
    answer_lower = entry["answer"].lower()
    for hedge in HEDGE_WORDS:
        if hedge in answer_lower:
            reasons.append(f"hedge word: '{hedge}'")
    for pattern in MARKUP_PATTERNS:
        if re.search(pattern, entry["answer"]):
            reasons.append(f"markup: {pattern}")
    return reasons


def rewrite_answer(question: str, bad_answer: str) -> str:
    """Ask the model to clean up ONE flagged answer -- same facts, plain language."""
    prompt = f"""Rewrite this answer to be plain, natural language.

Rules:
- Remove any documentation markup (backticks, :attr:, :func:, etc.) -- describe things in plain words instead
- Remove hedge words like "presumably", "likely", "probably" -- state the fact directly
- Keep the exact same technical meaning, don't add or remove information
- Keep it roughly the same length

Question: {question}
Original answer: {bad_answer}

Respond with ONLY the rewritten answer, no other text."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    with open(INPUT_PATH) as f:
        entries = [json.loads(line) for line in f]

    fixed_count = 0
    for entry in entries:
        reasons = flag_entry(entry)
        if reasons:
            print(f"\n[{entry['source_function']}] flagged for: {', '.join(reasons)}")
            print(f"  BEFORE: {entry['answer']}")

            entry["answer"] = rewrite_answer(entry["question"], entry["answer"])

            print(f"  AFTER:  {entry['answer']}")
            fixed_count += 1
            time.sleep(2.5)

    with open(INPUT_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*50}")
    print(f"Fixed {fixed_count} entries. File updated: {INPUT_PATH}")

    # Immediately re-check, using the SAME flag_entry function -- no
    # separate script needed to confirm the cleanup actually worked.
    print("\nRe-checking for anything still flagged...")
    remaining = sum(1 for e in entries if flag_entry(e))
    print(f"Remaining flagged entries: {remaining}/{len(entries)}")


if __name__ == "__main__":
    main()