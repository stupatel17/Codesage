import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"  # the strong model generating our training data

# How many functions to generate examples from. Starting small and
# deliberately -- verify quality before spending your rate limit on hundreds.
NUM_EXAMPLES = 20


def build_generation_prompt(function_entry: dict) -> str:
    """
    Construct the prompt that asks Groq to invent a realistic developer
    question about this specific function, grounded in its real code.
    """
    return f"""You are helping build a training dataset for a coding assistant.

Given this real function from the Flask web framework:

Function name: {function_entry['function_name']}
File: {function_entry['file']}

Docstring:
{function_entry['docstring']}

Source code:
{function_entry['source_code']}

Write ONE realistic question a developer might ask about this function
(e.g. "how do I use X", "why does X do Y", "what happens if I don't call X"),
and a clear, accurate, concise answer based ONLY on the code and docstring above.

Respond with ONLY valid JSON, no other text, in exactly this format:
{{"question": "...", "answer": "..."}}"""


def generate_qa_pair(function_entry: dict) -> dict | None:
    prompt = build_generation_prompt(function_entry)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw_text = response.choices[0].message.content

    try:
        parsed = json.loads(raw_text)
        return {
            "question": parsed["question"],
            "answer": parsed["answer"],
            "source_function": function_entry["function_name"],
            "source_file": function_entry["file"],
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Skipping {function_entry['function_name']} — bad response: {e}")
        return None


# ---------------------------------------------------------------
# Load extracted functions, generate one Q&A pair per function
# ---------------------------------------------------------------
with open("data/extracted_functions.json") as f:
    all_functions = json.load(f)

functions_to_use = all_functions[:NUM_EXAMPLES]

output_path = Path("data/synthetic_qa.jsonl")
generated_count = 0

with open(output_path, "w") as out_file:
    for i, func in enumerate(functions_to_use, start=1):
        print(f"[{i}/{len(functions_to_use)}] Generating for: {func['function_name']}")
        qa_pair = generate_qa_pair(func)

        if qa_pair:
            # Write immediately, one JSON object per line (JSONL format) --
            # if this script crashes on example 15, examples 1-14 are
            # already safely saved to disk, not lost.
            out_file.write(json.dumps(qa_pair) + "\n")
            out_file.flush()
            generated_count += 1

        # Free tier = 30 requests/minute = 1 every 2 seconds minimum.
        # We sleep a bit longer to stay safely under that limit.
        time.sleep(2.5)

print(f"\nDone. Generated {generated_count}/{len(functions_to_use)} examples.")
print(f"Saved to {output_path}")