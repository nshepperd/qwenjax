"""Exact policy gradient for multiple-choice answers with a stated confidence.

The policy is the frozen model behind a trainable cartridge. It is asked a
multiple-choice question and answers `B, 70% confidence`, with the confidence
a multiple of ten. With L letters that is L x 11 possible responses, and every
one of them is a fixed token string:

    <letter> , ␣ <d> ...        d = 0       -> "0%"        (0)
                                d = 2..9    -> "d0%"       (20..90)
                                d = 1, "0", then "%" -> 10,  then "0" -> 100

so the whole response distribution is three small softmaxes -- over the
letters, over the ten first digits given the letter, and over {"%", "0"} after
"10" -- and all of them come out of ONE teacher-forced forward of L rows,
`prompt + "X, 10"` for each letter X. With the distribution in hand the
expected reward

    J = sum_y pi(y) R(y),     R(l, c) = 2 [l = gold] - (c - [l = gold])^2

is a closed-form function of the cartridge, and its gradient is the quantity
GRPO estimates from samples -- here with no sampling, no baseline and no
variance (Rao-Blackwellised over the entire response space).

Each softmax is taken over the *valid* tokens only. The mass the unrestricted
model puts on them is reported (`format_mass`) but never trained: nothing here
pushes the model toward the answer format, only toward which letter and which
confidence, so a few steps cannot overfit the format itself.

Two things the first runs showed are needed (runs/mcq/run1..3):

* An abstain letter. With four options, +2 for the right one and nothing lost
  on a wrong one said at 0%, guessing is free -- and what transferred to
  free-form answers was exactly that: fewer abstentions, the same precision.
  With `abstain` set, the last letter means "what the question asks about is
  not in the codebase" and is simply the key of out-of-corpus questions, under
  the same reward. Optionally a wrong *answer* letter costs `wrong_penalty`
  (the abstain letter never does), so abstaining beats a guess the policy
  believes less than about 40%; at 1 it made no measurable difference.

* A supervised warm start for the confidence channel (`conf_sft_loss`). The
  base model says 100% with p = 0.9 and 50% with p < 1e-4; a policy gradient on
  a response scales with its probability, so low confidences never get signal.
  Yet the policy's own letter probability separates right from wrong well
  (AUROC .77). The warm start writes that certainty into the words: the target
  confidence for the letter the policy picks is the recalibrated probability
  that it is right, never below 1/L. (Targets for every letter, built from the
  reference policy, manufactured "E, 10%" once the letter choice moved.)
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

LETTERS = "ABCDE"
CONF_GRID = np.arange(0, 101, 10)  # percent; index c -> CONF_GRID[c]
N_CONF = len(CONF_GRID)
BONUS = 2.0


# -----------------------------------------------------------------------------
# reward and the outcome distribution
# -----------------------------------------------------------------------------


def reward_table(gold: Int[Array, "q"], n_letters: int = 4, *, bonus: float = BONUS,
                 wrong_penalty: float = 0.0, abstain: bool = False) -> Float[Array, "q L 11"]:
    """R(l, c) for every response: a flat bonus for the right letter, minus the
    Brier score of the stated confidence against whether that letter was right,
    minus `wrong_penalty` for a wrong letter other than the abstain letter."""
    letters = jnp.arange(n_letters)[None, :]
    correct = (letters == gold[:, None]).astype(jnp.float32)  # (q, L)
    conf = jnp.asarray(CONF_GRID, jnp.float32)[None, None, :] / 100.0
    r = bonus * correct[..., None] - (conf - correct[..., None]) ** 2
    if wrong_penalty:
        costed = (1.0 - correct) * ((letters != n_letters - 1) if abstain else 1.0)
        r = r - wrong_penalty * costed[..., None]
    return r


def conf_logprobs(digit_logits: Float[Array, "q L 10"], branch_logits: Float[Array, "q L 2"]) -> Float[Array, "q L 11"]:
    """log pi(confidence | letter). `branch_logits[..., 0]` is "%" after "10"
    (-> 10%), `[..., 1]` is "0" (-> 100%)."""
    ld = jax.nn.log_softmax(digit_logits, axis=-1)
    lb = jax.nn.log_softmax(branch_logits, axis=-1)
    return jnp.concatenate(
        [
            ld[..., 0:1],  # 0
            ld[..., 1:2] + lb[..., 0:1],  # 10
            ld[..., 2:10],  # 20..90
            ld[..., 1:2] + lb[..., 1:2],  # 100
        ],
        axis=-1,
    )


def outcome_logprobs(letter_logits: Float[Array, "q L"], digit_logits: Float[Array, "q L 10"],
                     branch_logits: Float[Array, "q L 2"]) -> Float[Array, "q L 11"]:
    """log pi(letter, confidence) from the three restricted softmaxes."""
    return jax.nn.log_softmax(letter_logits, axis=-1)[..., None] + conf_logprobs(digit_logits, branch_logits)


def expected_reward(logp: Float[Array, "q L 11"], reward: Float[Array, "q L 11"]) -> Float[Array, "q"]:
    return jnp.sum(jnp.exp(logp) * reward, axis=(-2, -1))


# -----------------------------------------------------------------------------
# rows
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnswerTokens:
    """Token ids of the answer format, checked against the tokenizer once."""

    letters: tuple[int, ...]
    digits: tuple[int, ...]
    comma: int
    space: int
    percent: int

    @classmethod
    def from_tokenizer(cls, tokenizer) -> AnswerTokens:
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        one = lambda s: enc(s)[0] if len(enc(s)) == 1 else None
        letters = tuple(one(x) for x in LETTERS)
        digits = tuple(one(str(d)) for d in range(10))
        t = cls(letters=letters, digits=digits, comma=one(","), space=one(" "), percent=one("%"))
        if None in letters + digits + (t.comma, t.space, t.percent):
            raise ValueError("answer format does not tokenize one token per symbol")
        # The enumeration assumes the full strings tokenize as the concatenation.
        for s, want in [("A, 70% confidence", [letters[0], t.comma, t.space, digits[7], digits[0], t.percent]),
                        ("E, 100% confidence", [letters[4], t.comma, t.space, digits[1], digits[0], digits[0], t.percent]),
                        ("B, 0% confidence", [letters[1], t.comma, t.space, digits[0], t.percent])]:
            if enc(s)[: len(want)] != want:
                raise ValueError(f"{s!r} tokenizes as {enc(s)}, expected prefix {want}")
        return t

    def tail(self, letter: int) -> list[int]:
        """`X, 10`: long enough to read all three decisions off one row."""
        return [self.letters[letter], self.comma, self.space, self.digits[1], self.digits[0]]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MCQBatch:
    """L rows per question, right-padded; `pos` are the three positions whose
    next-token logits are the letter, first-digit and 10-vs-100 decisions.
    The number of letters is `ref_logp.shape[1]`."""

    ids: Int[Array, "rows seq"]
    mask: Int[Array, "rows seq"]
    pos: Int[Array, "rows 3"]
    gold: Int[Array, "q"]
    ref_logp: Float[Array, "q L 11"]
    target: Int[Array, "q L"]  # warm start: grid index of the target confidence per letter
    weight: Float[Array, "q L"]  # warm start: weight of each letter's cross-entropy


def make_rows(prompts: list[list[int]], golds: list[int], toks: AnswerTokens, *, pad_id: int,
              n_letters: int = 4, pad_to: int = 64, ref_logp=None, target=None, weight=None) -> MCQBatch:
    rows, pos = [], []
    for p in prompts:
        for l in range(n_letters):
            rows.append(p + toks.tail(l))
            # logits at position i predict token i+1
            pos.append([len(p) - 1, len(p) + 2, len(p) + 4])
    width = -(-max(len(r) for r in rows) // pad_to) * pad_to
    ids = np.full((len(rows), width), pad_id, np.int32)
    mask = np.zeros((len(rows), width), np.int32)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = 1
    q = len(prompts)
    if ref_logp is None:
        ref_logp = np.zeros((q, n_letters, N_CONF), np.float32)
    if target is None:
        target = np.zeros((q, n_letters), np.int32)
    if weight is None:
        weight = np.zeros((q, n_letters), np.float32)
    return MCQBatch(ids=jnp.asarray(ids), mask=jnp.asarray(mask), pos=jnp.asarray(pos, jnp.int32),
                    gold=jnp.asarray(golds, jnp.int32), ref_logp=jnp.asarray(ref_logp, jnp.float32),
                    target=jnp.asarray(target, jnp.int32), weight=jnp.asarray(weight, jnp.float32))


# -----------------------------------------------------------------------------
# policy
# -----------------------------------------------------------------------------


def decision_logits(model, prefix, batch: MCQBatch, toks: AnswerTokens):
    """The three restricted logit sets, (q,L), (q,L,10), (q,L,2), and the format mass (q,3)."""
    hidden, _, _ = model.model(input_ids=batch.ids, attention_mask=batch.mask, prefix=prefix)
    h = jnp.take_along_axis(hidden, batch.pos[..., None], axis=1)  # (rows, 3, hidden)
    logits = model.get_lm_head()(h).astype(jnp.float32)  # (rows, 3, vocab)
    full = jax.nn.logsumexp(logits, axis=-1)  # (rows, 3)
    q, n = batch.gold.shape[0], batch.ref_logp.shape[1]
    letters = jnp.asarray(toks.letters[:n])
    digits = jnp.asarray(toks.digits)
    branch = jnp.asarray([toks.percent, toks.digits[0]])
    by_q = lambda x: x.reshape(q, n, *x.shape[1:])
    letter_logits = by_q(logits[:, 0][:, letters])[:, 0]  # identical across a question's rows
    digit_logits = by_q(logits[:, 1][:, digits])
    branch_logits = by_q(logits[:, 2][:, branch])
    mass = jnp.stack(
        [
            jnp.exp(jax.nn.logsumexp(letter_logits, -1) - by_q(full[:, 0])[:, 0]),
            jnp.mean(jnp.exp(jax.nn.logsumexp(digit_logits, -1) - by_q(full[:, 1])), -1),
            jnp.mean(jnp.exp(jax.nn.logsumexp(branch_logits, -1) - by_q(full[:, 2])), -1),
        ],
        axis=-1,
    )  # (q, 3)
    return (letter_logits, digit_logits, branch_logits), mass


def policy(model, prefix, batch: MCQBatch, toks: AnswerTokens):
    """(log pi over the L x 11 responses, format mass at the three decisions)."""
    logits, mass = decision_logits(model, prefix, batch, toks)
    return outcome_logprobs(*logits), mass


def mcq_loss(model, cartridge, batch: MCQBatch, beta, toks: AnswerTokens, *,
             normalize: bool = False, bonus: float = BONUS, conf_temp: float = 1.0,
             wrong_penalty: float = 0.0, abstain: bool = False):
    """-E[R] + beta * KL(pi || pi_ref), both exact over the L x 11 responses.

    `normalize` is GRPO's per-question standardisation, computed from the exact
    distribution instead of a sampled group: advantages (R - E R) / std R.

    `conf_temp` > 1 takes the expectation under a policy whose *confidence*
    softmaxes are flattened by that temperature (the analogue of rolling out at
    a high sampling temperature). It did not help (runs/mcq/run2-temp4): the
    real probability of a low confidence needs ~10 nats before the greedy
    answer changes. Kept for the record; the warm start is what works.
    """
    (ll, ld, lb), mass = decision_logits(model, cartridge.prefix(model.cache_dtype()), batch, toks)
    logp = outcome_logprobs(ll, ld, lb)
    r = reward_table(batch.gold, logp.shape[1], bonus=bonus, wrong_penalty=wrong_penalty, abstain=abstain)
    j = jnp.sum(jnp.exp(logp) * r, axis=(-2, -1))
    p = jnp.exp(outcome_logprobs(ll, ld / conf_temp, lb / conf_temp)) if conf_temp != 1.0 else jnp.exp(logp)
    if normalize:
        ps = jax.lax.stop_gradient(p)
        mean = jnp.sum(ps * r, axis=(-2, -1), keepdims=True)
        std = jnp.sqrt(jnp.sum(ps * (r - mean) ** 2, axis=(-2, -1), keepdims=True))
        objective = jnp.sum(p * (r - mean) / (std + 1e-3), axis=(-2, -1))
    else:
        objective = jnp.sum(p * r, axis=(-2, -1))
    kl = jnp.sum(jnp.exp(logp) * (logp - batch.ref_logp), axis=(-2, -1))
    loss = -jnp.mean(objective) + beta * jnp.mean(kl)
    return loss, (jnp.mean(j), jnp.mean(kl), jnp.mean(mass, axis=0))


def conf_sft_loss(model, cartridge, batch: MCQBatch, toks: AnswerTokens):
    """Cross-entropy of the confidence given each letter against `batch.target`,
    weighted by `batch.weight`. The letter softmax is not touched."""
    (ll, ld, lb), mass = decision_logits(model, cartridge.prefix(model.cache_dtype()), batch, toks)
    lc = conf_logprobs(ld, lb)  # (q, L, 11)
    ce = -jnp.take_along_axis(lc, batch.target[..., None], axis=-1)[..., 0]
    loss = jnp.sum(ce * batch.weight) / jnp.maximum(jnp.sum(batch.weight), 1e-6)
    logp = jax.nn.log_softmax(ll, -1)[..., None] + lc
    kl = jnp.sum(jnp.exp(logp) * (logp - batch.ref_logp), axis=(-2, -1))
    return loss, (loss, jnp.mean(kl), jnp.mean(mass, axis=0))


# -----------------------------------------------------------------------------
# recalibration (numpy): the warm start's targets
# -----------------------------------------------------------------------------


def isotonic_fit(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators: a non-decreasing step function of x fit to y.
    Returns (knots_x, knots_y) for `np.interp`."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order].astype(float), y[order].astype(float)
    vals, wts, los, his = [], [], [], []
    for xi, yi in zip(xs, ys):
        vals.append(yi); wts.append(1.0); los.append(xi); his.append(xi)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            lo = los[-2]
            for a in (vals, wts, los, his):
                a.pop()
            vals[-1], wts[-1], los[-1] = v, w, lo
            his[-1] = xi
    kx = np.array([v for pair in zip(los, his) for v in pair])
    ky = np.array([v for v in vals for _ in (0, 1)])
    return kx, ky


def conf_targets(letter_p: np.ndarray, knots) -> np.ndarray:
    """Grid index of the recalibrated probability that each letter is right."""
    cal = np.interp(letter_p, *knots)
    return np.clip(np.round(cal * 10), 0, 10).astype(np.int32)


# -----------------------------------------------------------------------------
# metrics (numpy, from the exact distribution)
# -----------------------------------------------------------------------------


def _ece(conf, hit):
    bins = np.clip((conf * 10).round().astype(int), 0, 10)
    return float(sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(11) if (bins == b).any()))


def _auroc(score, label):
    n1, n0 = label.sum(), (1 - label).sum()
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        ranks[m] = ranks[m].mean()
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarise(logp: np.ndarray, gold: np.ndarray, mass: np.ndarray | None = None, *,
              bonus: float = BONUS, wrong_penalty: float = 0.0, abstain: bool = False) -> dict:
    """What the policy would do, without sampling it.

    The greedy response is the most likely letter, then its most likely
    confidence; `acc`/`conf`/`ece`/`auroc` describe it. `exp_*` are expectations
    under the full distribution. With `abstain`, questions whose key is the last
    letter are out-of-corpus: `caught` is how often the greedy letter is the
    abstain letter there, `abstain_in` how often on real questions, and
    `selective_acc` the accuracy on the real questions actually answered.
    """
    p = np.exp(logp)  # (q, L, 11)
    q, n = len(gold), logp.shape[1]
    idx = np.arange(q)
    grid = CONF_GRID / 100.0
    pl = p.sum(-1)
    correct_tab = (np.arange(n)[None] == gold[:, None]).astype(np.float32)
    r = np.asarray(reward_table(jnp.asarray(gold), n, bonus=bonus, wrong_penalty=wrong_penalty, abstain=abstain))
    l_star = pl.argmax(-1)
    cond = p[idx, l_star] / np.maximum(pl[idx, l_star, None], 1e-30)
    conf = grid[cond.argmax(-1)]
    hit = (l_star == gold).astype(np.float32)
    out = dict(
        n=q, acc=float(hit.mean()), p_gold=float(pl[idx, gold].mean()),
        exp_reward=float((p * r).sum((-2, -1)).mean()),
        exp_brier=float((p * (grid[None, None] - correct_tab[..., None]) ** 2).sum((-2, -1)).mean()),
        modal_conf=float(conf.mean()), modal_brier=float(((conf - hit) ** 2).mean()),
        ece=_ece(conf, hit), auroc=_auroc(conf, hit), overconf=float(conf.mean() - hit.mean()),
        conf_when_right=float(conf[hit == 1].mean()) if hit.any() else float("nan"),
        conf_when_wrong=float(conf[hit == 0].mean()) if (hit == 0).any() else float("nan"),
        letter_marginal=[float(x) for x in pl.mean(0)],
    )
    if abstain:
        is_out = gold == n - 1
        ab = l_star == n - 1
        real = ~is_out
        answered = real & ~ab
        out.update(
            n_out=int(is_out.sum()),
            caught=float(ab[is_out].mean()) if is_out.any() else float("nan"),
            abstain_in=float(ab[real].mean()) if real.any() else float("nan"),
            acc_in=float(hit[real].mean()) if real.any() else float("nan"),
            selective_acc=float(hit[answered].mean()) if answered.any() else float("nan"),
            conf_answered=float(conf[answered].mean()) if answered.any() else float("nan"),
            auroc_answered=_auroc(conf[answered], hit[answered]) if answered.any() else float("nan"),
        )
    if mass is not None:
        out["format_mass"] = [float(x) for x in np.asarray(mass).mean(0)]
    return out


__all__ = ["BONUS", "CONF_GRID", "LETTERS", "AnswerTokens", "MCQBatch", "conf_logprobs", "conf_sft_loss",
           "conf_targets", "decision_logits", "expected_reward", "isotonic_fit", "make_rows", "mcq_loss",
           "outcome_logprobs", "policy", "reward_table", "summarise"]
