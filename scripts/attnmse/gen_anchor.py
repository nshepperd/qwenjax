"""Generate the anchor set: corpus-irrelevant conversations answered by the
init cartridge, used to pin off-corpus behaviour during training."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import json
import sys
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp

import cartridge as cli
from qwen_jax import chat

QUESTIONS = [
    "What's a good recipe for fluffy pancakes?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn rain.",
    "How do I reverse a linked list in Python?",
    "What causes a rainbow?",
    "Recommend three science fiction novels about first contact.",
    "What is the capital of Mongolia and what is it known for?",
    "How does compound interest work?",
    "Explain what a monad is in functional programming.",
    "What are some tips for improving sleep quality?",
    "How do vaccines work at a high level?",
    "What's the difference between a crocodile and an alligator?",
    "Write a short limerick about a cat who loves keyboards.",
    "How do I set up a git remote and push to it?",
    "What is the Riemann hypothesis about, roughly?",
    "Suggest a weekend itinerary for visiting Kyoto.",
    "Why is the sky blue during the day but red at sunset?",
    "What's a good beginner strength-training routine?",
    "Explain how a hash map handles collisions.",
    "What were the main causes of the fall of the Western Roman Empire?",
    "How do noise-cancelling headphones work?",
    "What is the difference between async and threading in Python?",
    "Give me a simple sourdough starter schedule.",
    "How far away is the Andromeda galaxy and how do we know?",
]

tokenizer = cli.load_tokenizer()
corpus = cli.load_corpus(tokenizer, None)
model = cli.load_model()
init = cli.init_cartridge(model, tokenizer, corpus, 1024, cli.DESCRIPTION)
im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)

rows = []
for i, q in enumerate(QUESTIONS):
    suffix = tokenizer.encode(chat.suffix([("user", q)], open_assistant=True),
                              add_special_tokens=False)
    out = model.generate(
        input_ids=jnp.asarray([suffix], dtype=jnp.int32),
        prefix=init.prefix(model.cache_dtype()),
        max_new_tokens=200, key=jax.random.key(i), temperature=0.7,
        stop_token_id=im_end, pad_token_id=im_end, progress_bar=False,
    )
    gen = out.tokens[0, len(suffix):].tolist()
    if im_end in gen:
        gen = gen[: gen.index(im_end)]
    rows.append({"user": q, "assistant": tokenizer.decode(gen)})
    print(f"[{i + 1}/{len(QUESTIONS)}] {q}", flush=True)

out_path = REPO / "runs/attnmse/overnight/anchor.jsonl"
out_path.write_text("\n".join(json.dumps(r) for r in rows))
print("wrote", out_path)
