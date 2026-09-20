"""The exact response distribution and its reward, checked against brute force on CPU."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from qwen_jax.mcq import (
    CONF_GRID, AnswerTokens, expected_reward, make_rows, outcome_logprobs, reward_table, summarise,
)


def _logits(seed, q=3):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    return (jax.random.normal(k1, (q, 4)) * 2, jax.random.normal(k2, (q, 4, 10)) * 2,
            jax.random.normal(k3, (q, 4, 2)) * 2)


def test_outcomes_are_a_distribution_and_match_the_token_tree():
    ll, ld, lb = _logits(0)
    logp = np.asarray(outcome_logprobs(ll, ld, lb))
    assert logp.shape == (3, 4, 11)
    np.testing.assert_allclose(np.exp(logp).sum((-2, -1)), 1.0, rtol=1e-5)
    pl, pd, pb = (np.asarray(jax.nn.softmax(x, -1)) for x in (ll, ld, lb))
    for q in range(3):
        for l in range(4):
            for c, pct in enumerate(CONF_GRID):
                if pct == 0:
                    want = pd[q, l, 0]
                elif pct == 10:
                    want = pd[q, l, 1] * pb[q, l, 0]
                elif pct == 100:
                    want = pd[q, l, 1] * pb[q, l, 1]
                else:
                    want = pd[q, l, pct // 10]
                np.testing.assert_allclose(np.exp(logp[q, l, c]), pl[q, l] * want, rtol=1e-5)


def test_reward_is_bonus_minus_brier_and_honesty_is_optimal():
    r = np.asarray(reward_table(jnp.asarray([2])))[0]
    assert r[2, 10] == pytest.approx(2.0)  # right, 100%
    assert r[2, 0] == pytest.approx(1.0)  # right, 0%: bonus 2, Brier 1
    assert r[0, 0] == pytest.approx(0.0)  # wrong, 0%
    assert r[0, 10] == pytest.approx(-1.0)  # wrong, 100%
    # With belief b that the chosen letter is right, the best stated confidence is b,
    # and a likelier letter is always worth more: picking the argmax is optimal.
    best = []
    for b in (0.2, 0.5, 0.8):
        ev = b * r[2] + (1 - b) * r[0]
        assert CONF_GRID[ev.argmax()] == pytest.approx(100 * b)
        best.append(ev.max())
    assert best == sorted(best)


def test_gradient_is_the_policy_gradient():
    ll, ld, lb = _logits(1, q=1)
    r = reward_table(jnp.asarray([1]))
    j = lambda a: expected_reward(outcome_logprobs(a, ld, lb), r)[0]
    g = np.asarray(jax.grad(j)(ll))[0]
    p = np.exp(np.asarray(outcome_logprobs(ll, ld, lb)))[0]
    pl = p.sum(-1)
    ql = (p * np.asarray(r)[0]).sum(-1) / pl
    np.testing.assert_allclose(g, pl * (ql - float(j(ll))), rtol=1e-4, atol=1e-6)


def test_summarise_reads_the_greedy_response():
    logp = np.full((2, 4, 11), -30.0, np.float32)
    logp[0, 1, 9] = 0.0  # B, 90%
    logp[1, 3, 2] = 0.0  # D, 20%
    s = summarise(logp, np.asarray([1, 0]))
    assert s["acc"] == 0.5 and s["modal_conf"] == pytest.approx(0.55)
    assert s["conf_when_right"] == pytest.approx(0.9) and s["conf_when_wrong"] == pytest.approx(0.2)
    assert s["exp_reward"] == pytest.approx(((2 - 0.01) + (-0.04)) / 2, abs=1e-4)


def test_rows_point_at_the_three_decisions():
    from cartridge import load_tokenizer

    tok = load_tokenizer()
    toks = AnswerTokens.from_tokenizer(tok)
    prompts = [tok.encode("hello there", add_special_tokens=False), tok.encode("x", add_special_tokens=False)]
    b = make_rows(prompts, [0, 3], toks, pad_id=tok.pad_token_id, pad_to=8)
    ids, pos = np.asarray(b.ids), np.asarray(b.pos)
    assert ids.shape[0] == 8 and ids.shape[1] % 8 == 0
    for row in range(8):
        letter = row % 4
        assert ids[row, pos[row, 0] + 1] == toks.letters[letter]
        assert ids[row, pos[row, 1] + 1] == toks.digits[1]  # the first confidence digit slot
        assert ids[row, pos[row, 2]] == toks.digits[0]  # "10" read; next is "%" or "0"
        assert np.asarray(b.mask)[row].sum() == pos[row, 2] + 1


def test_abstain_letter_makes_a_weak_guess_worse_than_abstaining():
    from qwen_jax.mcq import reward_table as rt

    r = np.asarray(rt(jnp.asarray([2, 4]), 5, wrong_penalty=1.0, abstain=True))
    real, out = r[0], r[1]
    assert real[2, 10] == pytest.approx(2.0) and real[0, 0] == pytest.approx(-1.0)  # wrong answer: -penalty
    assert real[4, 0] == pytest.approx(0.0)  # abstaining on a real question: no penalty
    assert out[4, 10] == pytest.approx(2.0) and out[1, 0] == pytest.approx(-1.0)
    # belief b in the best answer letter; abstaining at 0% is worth 0
    value = lambda b: max(b * real[2] + (1 - b) * real[0])
    assert value(0.3) < 0 < value(0.5)


def test_isotonic_recalibration_is_monotone_and_targets_land_on_the_grid():
    from qwen_jax.mcq import conf_targets, isotonic_fit

    rng = np.random.default_rng(0)
    x = rng.uniform(size=400)
    y = (rng.uniform(size=400) < x ** 2).astype(float)
    kx, ky = isotonic_fit(x, y)
    assert np.all(np.diff(kx) >= 0) and np.all(np.diff(ky) >= -1e-12)
    t = conf_targets(np.asarray([[0.01, 0.5, 0.99]]), (kx, ky))
    assert t.shape == (1, 3) and t[0, 0] <= t[0, 1] <= t[0, 2] and t.min() >= 0 and t.max() <= 10
    assert abs(t[0, 1] / 10 - 0.25) <= 0.15


def test_sft_targets_only_move_the_confidence_softmax():
    from qwen_jax.mcq import conf_logprobs

    ll, ld, lb = _logits(3, q=2)
    lc = np.asarray(conf_logprobs(ld, lb))
    np.testing.assert_allclose(np.exp(lc).sum(-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(outcome_logprobs(ll, ld, lb)),
                               np.asarray(jax.nn.log_softmax(ll, -1))[..., None] + lc, rtol=1e-5)


def test_summarise_with_abstain_splits_real_and_out_of_corpus():
    logp = np.full((3, 5, 11), -30.0, np.float32)
    logp[0, 1, 9] = 0.0  # real, answers B (right)
    logp[1, 4, 0] = 0.0  # real, abstains
    logp[2, 4, 10] = 0.0  # out-of-corpus, abstains: caught
    s = summarise(logp, np.asarray([1, 0, 4]), abstain=True, wrong_penalty=1.0)
    assert s["caught"] == 1.0 and s["abstain_in"] == 0.5 and s["selective_acc"] == 1.0 and s["acc_in"] == 0.5
