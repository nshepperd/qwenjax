"""Simple demo: load a Llama-architecture model (GGUF or safetensors) and continue a prompt.

Plain continuation, no chat template: suited to base models.
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a Llama-architecture model")
    parser.add_argument("config_path", help="HF model directory (config + tokenizer)")
    parser.add_argument("--gguf", default=None, help="Path to a llama-arch .gguf; without it, safetensors load from config_path")
    parser.add_argument("--prompt", default="The capital of France is", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--repeat", type=int, default=1, help="Generate this many times (later runs show compiled speed)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.config_path)
    t0 = time.perf_counter()
    if args.gguf is not None:
        from qwen_jax.llama import load_llama_gguf

        print(f"Loading GGUF from {args.gguf}...")
        model = load_llama_gguf(args.config_path, args.gguf)
    else:
        from qwen_jax.llama import load_llama_jax

        print(f"Loading safetensors from {args.config_path}...")
        model = load_llama_jax(args.config_path)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    prompt_len = input_ids.shape[1]
    for run in range(args.repeat):
        t0 = time.perf_counter()
        output = model.generate(
            input_ids=jnp.asarray(input_ids),
            max_new_tokens=args.max_tokens,
            key=jax.random.key(args.seed + run),
            temperature=args.temperature,
            stop_token_id=eos,
            pad_token_id=max(eos, 0),
            progress_bar=False,
        )
        tokens = output.tokens[0].tolist()
        elapsed = time.perf_counter() - t0
        if eos in tokens[prompt_len:]:
            tokens = tokens[: prompt_len + tokens[prompt_len:].index(eos)]

        # Decode prompt and continuation together: SentencePiece drops the
        # leading space of a sequence that starts mid-text.
        print(f"\n{tokenizer.decode(tokens, skip_special_tokens=True)}")
        note = "compile included" if run == 0 else "compiled"
        print(f"\n[{len(tokens) - prompt_len} tokens in {elapsed:.1f}s, {note}]")


if __name__ == "__main__":
    main()
