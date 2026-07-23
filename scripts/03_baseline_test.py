import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)


def generate_response(question: str, max_new_tokens: int = 150) -> str:
    """Send one question through the model, return only the new text it generated."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Domain questions -- these represent what CodeSage should eventually be
# good at answering. We'll replace/expand these later once Phase 6's RAG
# layer points at a real codebase.
baseline_questions = [
    "What is a race condition in programming? Explain in two sentences.",
    "What's the difference between a list and a tuple in Python?",
    "Write a Python function that checks if a string is a palindrome.",
    "What does the @property decorator do in Python?",
]

results = []
for i, question in enumerate(baseline_questions, start=1):
    print(f"[{i}/{len(baseline_questions)}] Asking: {question}")
    answer = generate_response(question)
    print(f"Answer: {answer}\n")
    results.append({"question": question, "baseline_answer": answer})

# Save results so later phases (LoRA, DPO) can be compared against this
output_path = Path("results/baseline_outputs.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} baseline answers to {output_path}")