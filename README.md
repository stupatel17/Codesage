# CodeSage
A fine-tuned, aligned, retrieval-augmented code assistant — built step by step
to learn the full modern LLM pipeline: fine-tuning, LoRA/QLoRA, DPO alignment,
and RAG.

## Status
Just getting started — Phase 1 in progress.

## Progress Log

### Phase 1 — Baseline
- Loaded Qwen2.5-0.5B-Instruct via Hugging Face `transformers`
- Confirmed correct prompt formatting via `apply_chat_template`
  (Qwen expects `<|im_start|>role ... <|im_end|>` structure)
- Ran a fixed set of domain questions, saved outputs to `results/baseline_outputs.json`
- Every later phase gets compared back against this same baseline file
- Observation: base model doesn't reliably follow explicit constraints
  (e.g. "explain in two sentences" produced a full paragraph) — a concrete
  target for improvement once we get to fine-tuning/DPO