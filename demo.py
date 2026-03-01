"""Simple demo: load a Qwen3-VL model and generate text."""

import argparse

import jax
import torch
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from qwen_jax.loading import load_qwen3_jax
from qwen_jax.testutil import to_jax

# Limit CUDA memory for the HF processor/tokenizer
torch.cuda.memory.set_per_process_memory_fraction(0.2)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Qwen3-VL")
    parser.add_argument("model_path", help="Path to Qwen3-VL model directory")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Text prompt (default: 'Hello, how are you?')")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    processor: Qwen3VLProcessor = Qwen3VLProcessor.from_pretrained(args.model_path)
    model = load_qwen3_jax(args.model_path)

    # Format as a chat message
    chat_prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    inputs = processor(
        text=[chat_prompt],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}

    # Generate
    key = jax.random.key(args.seed)
    im_end_token = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        key=key,
        temperature=args.temperature,
        stop_token_id=im_end_token,
        pad_token_id=im_end_token,
    )

    # Extract and decode generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = output.tokens[0, prompt_len:].tolist()

    if im_end_token in gen_tokens:
        gen_tokens = gen_tokens[: gen_tokens.index(im_end_token)]

    response = processor.tokenizer.decode(gen_tokens, skip_special_tokens=False)
    print(f"\n{response}")


if __name__ == "__main__":
    main()
