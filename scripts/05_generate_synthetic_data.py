import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
QUESTIONS_PER_FUNCTION = 2
OUTPUT_PATH = Path("data/synthetic_qa.jsonl")


def build_generation_prompt(function_entry: dict) -> str:
    return f"""You are helping build a training dataset for a coding assistant.

Given this real function from the Flask web framework:

Function name: {function_entry['function_name']}
File: {function_entry['file']}

Docstring:
{function_entry['docstring']}

Source code:
{function_entry['source_code']}

Write {QUESTIONS_PER_FUNCTION} DIFFERENT realistic questions a developer might ask
about this function -- covering different angles (e.g. one "how do I use X",
one "what happens if X fails/misused", one "why does X exist" -- pick whichever
angles genuinely make sense for this specific function). For each question, give
a clear, accurate, concise answer based ONLY on the code and docstring above.
Do not use documentation markup (no backticks, no :attr:/:func: syntax) --
plain natural language only. Do not hedge with words like "presumably" or
"probably" -- state facts directly.

Respond with ONLY a valid JSON array, no other text, in exactly this format:
[{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}]"""


def parse_wait_seconds(error_message: str, default: int = 360) -> int:
    """
    Groq's rate-limit error message includes a suggested wait time like
    'Please try again in 5m57.696s'. Parse that out instead of guessing,
    so we wait exactly as long as needed and not a second longer.
    """
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_message)
    if not match:
        return default
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return int(minutes * 60 + seconds) + 5  # small safety buffer


def generate_qa_pairs(function_entry: dict, max_retries: int = 5) -> list[dict]:
    prompt = build_generation_prompt(function_entry)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            parsed = json.loads(response.choices[0].message.content)
            return [
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "source_function": function_entry["function_name"],
                    "source_file": function_entry["file"],
                }
                for item in parsed
            ]
        except RateLimitError as e:
            wait = parse_wait_seconds(str(e))
            print(f"  Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Skipping {function_entry['function_name']} — bad response: {e}")
            return []

    print(f"  Giving up on {function_entry['function_name']} after {max_retries} rate-limit retries.")
    return []


# ---------------------------------------------------------------
# RESUME SUPPORT: figure out which functions already have saved
# output, so re-running this script never wastes tokens redoing work.
# ---------------------------------------------------------------
processed_keys = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        for line in f:
            entry = json.loads(line)
            processed_keys.add((entry["source_function"], entry["source_file"]))
    print(f"Resuming: {len(processed_keys)} functions already have saved output, skipping them.\n")

with open("data/extracted_functions.json") as f:
    all_functions = json.load(f)

remaining = [
    func for func in all_functions
    if (func["function_name"], func["file"]) not in processed_keys
]
print(f"{len(remaining)} functions remaining to process.\n")

generated_count = 0
with open(OUTPUT_PATH, "a") as out_file:  # APPEND, never overwrite what's already saved
    for i, func in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] Generating for: {func['function_name']} ({func['file']})")
        qa_pairs = generate_qa_pairs(func)

        for pair in qa_pairs:
            out_file.write(json.dumps(pair) + "\n")
            out_file.flush()
            generated_count += 1

        time.sleep(2.5)

print(f"\nDone. Generated {generated_count} new examples this run.")
print(f"Total examples in file: {len(processed_keys) + generated_count} (approx, if none were skipped for bad responses)")