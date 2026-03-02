"""Demo: vision understanding with Qwen3-VL in JAX."""

import argparse
from pathlib import Path

import jax
import torch
from PIL import Image
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from qwen_jax.loading import load_qwen3_jax
from qwen_jax.testutil import to_jax

# Limit CUDA memory for the HF processor/tokenizer
torch.cuda.memory.set_per_process_memory_fraction(0.2)

DEFAULT_IMAGE = "https://eepylomf.net/images/eepy.jpg"
MAX_IMAGE_PIXELS = 1024 * 1024  # ~1 megapixel


def load_image(source: str, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """Load an image from a local path or URL, resizing if too large."""
    if source.startswith(("http://", "https://")):
        import urllib.request
        from io import BytesIO

        with urllib.request.urlopen(source) as resp:
            img = Image.open(BytesIO(resp.read())).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")

    # Resize if the image exceeds max_pixels, preserving aspect ratio
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def main():
    parser = argparse.ArgumentParser(description="Vision understanding with Qwen3-VL")
    parser.add_argument("model_path", help="Path to Qwen3-VL model directory")
    parser.add_argument("images", nargs="*", default=[DEFAULT_IMAGE], help="Image paths or URLs (default: eepy.jpg)")
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Question about the image(s) (default: 'Describe this image in detail.')",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load images
    print(f"Loading {len(args.images)} image(s)...")
    images = [load_image(src) for src in args.images]
    for i, (src, img) in enumerate(zip(args.images, images)):
        print(f"  [{i+1}] {Path(src).name if not src.startswith('http') else src[:80]} ({img.size[0]}x{img.size[1]})")

    # Load model
    print(f"Loading model from {args.model_path}...")
    processor: Qwen3VLProcessor = Qwen3VLProcessor.from_pretrained(args.model_path)
    model = load_qwen3_jax(args.model_path)

    # Build chat message with image placeholders and text prompt
    content = [{"type": "image", "url": None} for _ in images]
    content.append({"type": "text", "text": args.prompt})

    chat_prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
    )

    print(f"\nFormatted prompt:\n{chat_prompt}\n")

    # Tokenize with images
    inputs = processor(
        text=chat_prompt,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}

    # Generate
    key = jax.random.key(args.seed)
    im_end_token = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    print("Generating...")
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
