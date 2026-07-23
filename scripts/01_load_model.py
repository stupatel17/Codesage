from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded.")

print("Loading model (this downloads ~1GB the first time)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
print("Model loaded.")

num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")