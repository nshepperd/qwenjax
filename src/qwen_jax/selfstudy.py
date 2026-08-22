"""Self-study: the model writes its own training data about a corpus.

From arXiv 2506.06266 section 4.1. A random chunk of the corpus goes in the
system prompt; a generic seed prompt asks the model to start a conversation
about it; the model answers its own message, again with the chunk in context.
The (user, assistant) pair is a training example, and the chunk that produced
it is what the teacher sees when the pair is distilled into a cartridge.

Two knobs shape the data distribution and both matter (section 5.3):

- Chunking. Short windows (512-4096 tokens) make the model attend to one part
  of the corpus at a time, and let a corpus longer than the context window be
  covered piecewise.
- Seed prompts. Five families -- structuring, summarisation, question, use
  case, creative -- none mentioning the corpus's subject. Diversity here is
  worth several accuracy points over a single prompt.

Generation runs in batches through `model.generate`; prompts are left-padded
and bucketed to a multiple of `pad_to` so a run compiles a handful of shapes,
not one per batch.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import chat

# -----------------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Corpus:
    """A corpus as one token stream, with the text it came from."""

    text: str
    ids: np.ndarray  # int32, (n,)

    @classmethod
    def from_text(cls, tokenizer, text: str) -> Corpus:
        ids = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        return cls(text=text, ids=ids)

    @classmethod
    def from_files(cls, tokenizer, paths: list[str | Path], root: str | Path | None = None) -> Corpus:
        """Concatenate files, each under a `### <relative path>` header.

        The header is what lets a chunk say which file it is from, and lets
        questions about "the X module" resolve.
        """
        root = Path(root) if root is not None else None
        parts = []
        for p in paths:
            p = Path(p)
            name = str(p.relative_to(root)) if root is not None else str(p)
            parts.append(f"### {name}\n{p.read_text()}\n\n")
        return cls.from_text(tokenizer, "".join(parts))

    def __len__(self) -> int:
        return len(self.ids)

    def head(self, n: int) -> np.ndarray:
        return self.ids[:n]

    def sample_chunk(self, rng: random.Random, min_tokens: int, max_tokens: int) -> np.ndarray:
        """A uniformly random token window of random length in [min, max]."""
        n = min(rng.randint(min_tokens, max_tokens), len(self.ids))
        start = rng.randint(0, len(self.ids) - n)
        return self.ids[start:start + n]


# -----------------------------------------------------------------------------
# Seed prompts
# -----------------------------------------------------------------------------

# Generic by design: nothing here knows what the corpus is about.
SEED_PROMPTS: dict[str, list[str]] = {
    "structuring": [
        "Please start a conversation by asking me to restructure a part of the "
        "document above into a table, list, or outline. Say exactly which part.",
        "Begin by asking me to reorganise some specific piece of the document above "
        "into a more structured form (a table, a numbered list, a tree). Be specific "
        "about which piece.",
    ],
    "summarization": [
        "Please start a conversation by asking me to summarise a specific section "
        "of the document above.",
        "Begin by asking me for a short summary of one particular part of the "
        "document above, naming the part.",
    ],
    "question": [
        "Please start a conversation by asking me a specific, detailed question "
        "about the document above -- one that can be answered from it.",
        "Ask me one precise factual question whose answer is in the document above. "
        "Do not answer it yourself.",
        "Ask me a question that requires understanding how two parts of the document "
        "above relate to each other.",
    ],
    "use_case": [
        "Please start a conversation by asking me how the material in the document "
        "above could be used or applied to a concrete task.",
        "Begin by asking me a practical question: how would someone use what is "
        "described in the document above to get something done?",
    ],
    "creative": [
        "Please start a conversation by asking me to explain something from the "
        "document above by analogy, or to a particular audience.",
        "Begin by asking me to critique, compare, or propose an alternative to "
        "something described in the document above.",
    ],
}


def sample_seed(rng: random.Random) -> tuple[str, str]:
    kind = rng.choice(sorted(SEED_PROMPTS))
    return kind, rng.choice(SEED_PROMPTS[kind])


# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Example:
    """One self-study conversation and the chunk it was generated from."""

    chunk_ids: list[int]
    user: str
    assistant: str
    seed_kind: str = ""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Example:
        return cls(**json.loads(s))


def save_examples(examples: list[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def load_examples(path: str | Path) -> list[Example]:
    with open(path) as f:
        return [Example.from_json(line) for line in f if line.strip()]


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def _left_pad(batch: list[list[int]], pad_id: int, pad_to: int) -> tuple[np.ndarray, np.ndarray]:
    width = -(-max(len(b) for b in batch) // pad_to) * pad_to
    ids = np.full((len(batch), width), pad_id, dtype=np.int32)
    mask = np.zeros((len(batch), width), dtype=np.int32)
    for i, b in enumerate(batch):
        ids[i, width - len(b):] = b
        mask[i, width - len(b):] = 1
    return ids, mask


def generate_batch(
    model,
    tokenizer,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    key,
    temperature: float,
    pad_to: int = 256,
) -> list[str]:
    """Sample a completion for each token prompt; returns decoded text up to <|im_end|>."""
    im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else im_end
    ids, mask = _left_pad(prompts, pad_id, pad_to)
    out = model.generate(
        input_ids=jnp.asarray(ids),
        attention_mask=jnp.asarray(mask),
        max_new_tokens=max_new_tokens,
        key=key,
        temperature=temperature,
        stop_token_id=im_end,
        pad_token_id=im_end,
        progress_bar=False,
    )
    gen = np.asarray(out.tokens[:, ids.shape[1]:])
    texts = []
    for row in gen:
        row = row.tolist()
        if im_end in row:
            row = row[: row.index(im_end)]
        texts.append(tokenizer.decode(row, skip_special_tokens=False).strip())
    return texts


def self_study(
    model,
    tokenizer,
    corpus: Corpus,
    *,
    n: int,
    description: str,
    key,
    seed: int = 0,
    batch_size: int = 8,
    chunk_tokens: tuple[int, int] = (512, 2048),
    max_user_tokens: int = 128,
    max_assistant_tokens: int = 384,
    temperature: float = 0.7,
    pad_to: int = 256,
    progress=None,
) -> list[Example]:
    """Generate `n` conversations (Algorithm 1 of the paper, with k = 1).

    Each example is two generations: participant A, with the chunk and a seed
    prompt in context, writes the user message; participant B, with only the
    chunk in context, writes the reply. Both are the same model.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    while len(examples) < n:
        b = min(batch_size, n - len(examples))
        chunks = [corpus.sample_chunk(rng, *chunk_tokens) for _ in range(b)]
        seeds = [sample_seed(rng) for _ in range(b)]
        chunk_texts = [tokenizer.decode(c, skip_special_tokens=False) for c in chunks]
        systems = [f"{description}\n\n{t}" if description else t for t in chunk_texts]

        # A: ask something about the chunk.
        key, k1, k2 = jax.random.split(key, 3)
        prompts = [
            chat.encode(tokenizer, sys, [("user", seed_text)], open_assistant=True).ids
            for sys, (_, seed_text) in zip(systems, seeds)
        ]
        users = generate_batch(model, tokenizer, prompts, max_new_tokens=max_user_tokens,
                               key=k1, temperature=temperature, pad_to=pad_to)

        # B: answer it, with the chunk but without the seed prompt.
        prompts = [
            chat.encode(tokenizer, sys, [("user", u)], open_assistant=True).ids
            for sys, u in zip(systems, users)
        ]
        answers = generate_batch(model, tokenizer, prompts, max_new_tokens=max_assistant_tokens,
                                 key=k2, temperature=temperature, pad_to=pad_to)

        for c, (kind, _), u, a in zip(chunks, seeds, users, answers):
            if u and a:
                examples.append(Example(chunk_ids=c.tolist(), user=u, assistant=a, seed_kind=kind))
        if progress is not None:
            progress(len(examples), n)
    return examples[:n]


__all__ = [
    "SEED_PROMPTS",
    "Corpus",
    "Example",
    "generate_batch",
    "load_examples",
    "sample_seed",
    "save_examples",
    "self_study",
]
