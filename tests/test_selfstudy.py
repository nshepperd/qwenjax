"""Pointed self-study, with generation stubbed: the schedule, marking, anchoring
and the two examples per conversation are all CPU logic."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from qwen_jax import selfstudy
from qwen_jax.selfstudy import (
    MARK_CLOSE, MARK_OPEN, Corpus, Example, corpus_spans, is_anchored, pointed_self_study,
    span_anchors,
)

SRC = """### a.py
def rotate_half(x, axis=-1):
    return x

class KVCache:
    keys: int
    def from_cache(self, cache, batch_index):
        return cache

# ---------------- filler that names nothing ----------------
"""


@pytest.fixture(scope="module")
def tokenizer():
    from cartridge import load_tokenizer

    return load_tokenizer()


def test_spans_cover_corpus():
    spans = corpus_spans(103, 40)
    assert spans == [(0, 40), (40, 80), (80, 103)]


def test_anchors_and_anchoring():
    a = span_anchors("def from_cache(self, cache, batch_index):\n    return cache")
    assert "from_cache" in a and "batch_index" in a
    assert "self" not in a and "return" not in a and "def" not in a  # stopped / too short
    assert is_anchored("What does from_cache do?", a)
    assert not is_anchored("What does from_cache_v2 do?", {"from_cache"})
    assert not is_anchored("nothing named here", a)


def test_pointed_self_study_stubbed(tokenizer, monkeypatch):
    corpus = Corpus.from_text(tokenizer, SRC * 3)
    spans = corpus_spans(len(corpus), 24)
    calls = {"ask": 0, "answer": 0}
    seen_marked = []

    def fake_generate(model, tok, prompts, *, max_new_tokens, key, temperature, pad_to=256):
        out = []
        for p in prompts:
            text = tok.decode(p, skip_special_tokens=False)
            if MARK_OPEN in text:
                calls["ask"] += 1
                marked = text[text.index(MARK_OPEN) + len(MARK_OPEN): text.index(MARK_CLOSE)]
                seen_marked.append(marked)
                anchors = sorted(span_anchors(marked))
                # first ask of the run is unanchored, to exercise the retry
                if calls["ask"] == 1 or not anchors:
                    out.append("Could you explain the marked bit?")
                else:
                    out.append(f"Could you walk me through {anchors[0]}?")
            else:
                calls["answer"] += 1
                assert MARK_OPEN not in text and MARK_CLOSE not in text
                out.append("Sure: it does the thing.")
        return out

    monkeypatch.setattr(selfstudy, "generate_batch", fake_generate)
    examples, stats = pointed_self_study(
        None, tokenizer, corpus, spans=spans, description="desc", key=jax.random.key(0),
        batch_size=4, chunk_tokens=(30, 60), tries=2,
    )
    anchorable = [sp for sp in spans if span_anchors(tokenizer.decode(corpus.ids[sp[0]:sp[1]]))]
    assert stats["spans"] == len(spans)
    assert stats["unanchorable"] == len(spans) - len(anchorable)
    assert stats["rejected"] >= 1 and stats["given_up"] == 0
    assert len(examples) == 2 * len(anchorable)

    by_span = {}
    for ex in examples:
        by_span.setdefault(tuple(ex.span), []).append(ex)
    for (s, e), pair in by_span.items():
        kinds = sorted(ex.seed_kind.split(":")[0] for ex in pair)
        assert kinds == ["ask", "pointed"]
        ans = next(ex for ex in pair if ex.seed_kind.startswith("pointed"))
        ask = next(ex for ex in pair if ex.seed_kind.startswith("ask"))
        # same unmarked chunk, containing the span
        assert ans.chunk_ids == ask.chunk_ids
        chunk = np.asarray(ans.chunk_ids)
        assert MARK_OPEN not in tokenizer.decode(chunk)
        span_ids = corpus.ids[s:e].tolist()
        assert any(chunk[i:i + len(span_ids)].tolist() == span_ids
                   for i in range(len(chunk) - len(span_ids) + 1))
        # the question names something from the span; it is the ask example's target
        assert is_anchored(ans.user, span_anchors(tokenizer.decode(corpus.ids[s:e])))
        assert ask.assistant == ans.user
        assert "marked" not in ask.user.lower() and "document" not in ask.user.lower()
        assert "marked" not in ans.user.lower()
    # the marked text handed to the asker was exactly the span
    assert any(m.strip() == tokenizer.decode(corpus.ids[s:e]).strip()
               for m in seen_marked for (s, e) in spans)

    # round trip keeps the span
    ex = Example.from_json(examples[0].to_json())
    assert ex.span == examples[0].span


def test_give_up_after_tries(tokenizer, monkeypatch):
    corpus = Corpus.from_text(tokenizer, SRC)
    spans = corpus_spans(len(corpus), 24)
    n_ask = {"n": 0}

    def never_anchored(model, tok, prompts, **kw):
        n_ask["n"] += len(prompts)
        return ["no names here" for _ in prompts]

    monkeypatch.setattr(selfstudy, "generate_batch", never_anchored)
    examples, stats = pointed_self_study(
        None, tokenizer, corpus, spans=spans, description="d", key=jax.random.key(0),
        batch_size=8, chunk_tokens=(20, 40), tries=3,
    )
    assert examples == []
    anchorable = stats["spans"] - stats["unanchorable"]
    assert stats["given_up"] == anchorable
    assert n_ask["n"] == 3 * anchorable
