from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

question = "What is a race condition in programming? Explain in two sentences."

# Instruct models expect a specific chat structure, not raw text --
# this wraps our question the way Qwen was actually trained to receive it.
messages = [
    {"role": "user", "content": question}
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("--- Actual prompt sent to the model ---")
print(prompt)
print("----------------------------------------")

inputs = tokenizer(prompt, return_tensors="pt")

output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
)


new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n--- Model's response ---")
print(response)