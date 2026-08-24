"""Simple demo: load a Qwen3.5 model (GGUF or safetensors) and generate text."""
from __future__ import annotations

import argparse

import jax
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with Qwen3.5")
    parser.add_argument("config_path", help="HF model directory (config + tokenizer)")
    parser.add_argument("--gguf", default=None, help="Path to a qwen35-arch .gguf; without it, safetensors load from config_path")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.config_path)
    if args.gguf is not None:
        from qwen_jax.qwen35 import load_qwen35_gguf

        print(f"Loading GGUF from {args.gguf}...")
        model = load_qwen35_gguf(args.config_path, args.gguf)
    else:
        from qwen_jax.qwen35 import load_qwen35_jax

        print(f"Loading safetensors from {args.config_path}...")
        model = load_qwen35_jax(args.config_path)

    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="np", return_dict=True
    )["input_ids"]

    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    output = model.generate(
        input_ids=jax.numpy.asarray(input_ids),
        max_new_tokens=args.max_tokens,
        key=jax.random.key(args.seed),
        temperature=args.temperature,
        stop_token_id=im_end_token,
        pad_token_id=im_end_token,
    )

    prompt_len = input_ids.shape[1]
    gen_tokens = output.tokens[0, prompt_len:].tolist()
    if im_end_token in gen_tokens:
        gen_tokens = gen_tokens[: gen_tokens.index(im_end_token)]

    response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    print(f"\n{response}")


if __name__ == "__main__":
    main()
