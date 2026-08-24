# Cartridges literature, read against two plans (2026-08-23)

Plans: (1) **yonaka** — nightly "dream" compresses the day's context into a cartridge,
recursively consolidated over days so the distant past is vague and recurrent things
stay clear; (2) **starchart recall** — cartridges over the board, so cards light up
from natural-language or thinking-block queries.

Papers (all IDs verified; two opus readers, full HTML text):

- **CAS** — Cartridges at Scale, Hardalov/Iglesias/de Gispert, Amazon AGI, arXiv 2606.04557.
- **Diaz** — Learned Structure in Cartridges, arXiv 2508.17032 (MechInterp workshop, single author, admittedly 4–6× under-trained).
- **Still** — Amortized KV Cache Compaction in a Single Forward Pass, Baseten, arXiv 2606.07878.
- **CDLM** — Context Distillation as Latent Memory Management, CUHK/Huawei, arXiv 2605.28889 (LoRA adapters, not KV).
- **Cartridges @ ICLR 2026** (poster; arXiv 2506.06266 v3). Not retitled. Engram = Eyuboglu/Biderman startup, $98M, nothing technical public. (Unrelated: DeepSeek "Engram", Jan 2026.)

Our own datum: two cartridges trained in isolation (qwen-jax, tetris; p=1024; Qwen3-VL-8B
Q4_K_M) composed with RoPE repositioning answer 11/12 questions from the right one;
~24 heads in layers 11–23 route between them, invisible in the all-head average
(`runs/compose/`, card 1fhbc1).

## 1. What is actually established about composition

| Source | N | Docs | Training | Result |
|---|---|---|---|---|
| Cartridges §5.4 | 2 | 10-Ks, ~100k tok each | isolated | beats truncated ICL and single cartridge; order-insensitive *because no positional encoding is applied* |
| CAS Table 3 | 20 | LongHealth patients, near-identical format, 20× | isolated | **26.0** (below the 37.5 no-context floor; oracle 73.6) |
| CAS Table 3 | 20 | same | joint, 20 active distractors | **77.8** (oracle 79.0) |
| CAS Table 3 | 20 | same | joint, 10 active | 56.4 — train-time k must match deploy-time k |
| ours | 2 | disjoint domains | isolated | 11/12 |

CAS establishes collapse at exactly one point (N=20, near-duplicate docs, 5-way MC) and
has no N=2..10 ablation and no similarity ablation. Our result sits at the easy corner
(N=2, disjoint vocabularies, open questions). **Neither refutes the other; the untested
axes are N, document similarity and compression ratio.**

Mechanism, assembled from Diaz + CAS (hypothesis, not stated by either): keys barely move
from init and act as the retrieval router; values carry the content. Shared-init
isolation-trained cartridges therefore have near-identical routers → the model cannot
select → collapse. Cart-specific init (CAS: ~50% lower initial loss, ~10% lower final
loss; ICLR paper: 55.3 vs 29.9 random vs 51.3 *wrong-corpus* init) is what makes
cartridges routable at all. Ours are initialised from their own corpora with distinct
description headers — maximally distinct routers — which is the likely reason they route.
Diaz's "freeze keys entirely" proposal is therefore *bad* for multi-cartridge use.

Two composition operators exist and we use the non-canonical one: Cartridges/CAS
concatenate without repositioning (exact order invariance: CAS ordered = shuffled =
77.8); our `compose(reposition=True)` re-rotates each piece to its slot, so order matters
and a cartridge is evaluated at positions it never trained at. Still independently
found that compacting *rotated* keys is unstable and un-rotates first. Settle it:
same 12 questions, reposition on/off, both orders.

## 2. Recipe deltas worth adopting now (cheap, each measured)

- **Mixed-visibility training** (CAS): with prob 0.25 add k~U(1,10) distractor
  cartridges to the prefix during distillation; k_max ≥ number loaded at inference.
  Same step count; cost is the longer prefix per step. Improves the *oracle* too
  (73.6 → 79.0).
- **Data > architecture** (CAS Table 6, FinQA): questions from a larger, different
  model + 20 questions per call: +6.8; proportional-to-length doc sampling: +2.0.
  Answers/log-probs must still come from the student's own teacher pass. 10–20 epochs
  reach 95% of 80-epoch quality. Audit the synthetic data: 59% of numbers in CAS's
  FinQA data were not in the source, 9% fabricated; entity↔value bindings get lost.
- **Optimiser** (CAS): peak LR 0.05–0.1 (ours 3e-2, in range), slow linear decay to
  0.02× (+4.2), fp32 Adam moments with bf16 weights (we keep params fp32 already), per-
  cartridge 20-step warmup when several train at once (+1.8).
- **Init**: Sampled Chunk Init (Diaz): p/64 random 64-token chunks of the corpus,
  concatenated, KV of that; significantly faster convergence (p<0.05). For a day of
  conversation first-k = the greeting, so SCI matters more there than for code.
- **Teacher-cache reuse** (CDLM): prefill the teacher context once per corpus, reuse
  across steps → 8.4× training speedup, training time ~constant in context length.
  Check whether `distill.py` recomputes the teacher prefix every step; if so this is the
  highest-value/lowest-risk change available.
- **Compression tolerance is content-dependent** (CAS Table 2): prose/narrative 81.1
  @2× → 77.3 @100× (graceful); tables/numbers 62.7 @2× → 23.0 @100×; extractive tech
  docs lossless to 100× and *above* raw context. Dates, IDs, dosages, commitments do not
  survive compression.

## 3. Yonaka

- **One cartridge per day/episode, composed at read time; never re-distil day N+1 on
  top of day N.** CDLM Appendix A: the ideal fixed point of cumulative distillation is
  the base model reading the whole concatenated history, so it inherits long-context
  degradation and absorbs noise before forgetting even enters; their cumulative
  baselines are near-degenerate. Still's iterative compaction is the closest direct test
  of recursion: an 8-merge-trained compactor collapses to 1.5% after 128 merges (below
  no-context) despite a *larger* retained cache; needle recall goes to ~0 while gist
  tasks stay 73–90%.
- **Train dreams jointly from day one** (CAS): every nightly distillation samples recent
  cartridges into the prefix; decide the deployed load (e.g. 7 recent + 3 retrieved) before
  the first dream and pin k_max to it. Mismatch cost is 21 points.
- **"Vague distant past" is nearly free**: a compression ladder (today 2×, week 10×,
  month 50×, year 100×) is CAS's Table 2; narrative degrades 3.8 points over 50×.
  **"Important/recurrent things clearer" is not in any paper** — Still's recurrence is
  uniform-lossy and its only weighting (w ∝ N−p+1) is for stability. That is the open
  design problem. Consolidation should be *re-distillation* over the month's transcripts
  (or over its cartridges used as teacher context) into a smaller cartridge, not tensor
  merging (CAS "document grouping": shared vocabulary lets the budget go to unique info).
- **Extract hard facts to text/starchart in the dream, let the cartridge carry texture.**
  FinQA column.
- **Prefix-tuning preserves generality, LoRA does not** (ICLR ablation: MMLU 54.7→54.3 as
  cartridge grows to 0.96 GB vs 54.7→45.3 for LoRA). For a persistent character this is
  the difference between remembering yesterday and being lobotomised by it.
- **Mid-conversation loading is unsolved in CAS** ("would invalidate previously computed
  KV entries"). Fine if memories load at wake-up. Our repositioning is exactly the tool
  for inserting a cartridge at an arbitrary offset without recomputing it — a real
  contribution direction if mid-conversation recall is wanted.
- Cartridges' authors describe the plan almost verbatim in the ICLR rebuttal: "one
  cartridge per user… updated nightly".

## 4. Starchart recall

- **CAS names a learned cartridge retriever as its top open problem** ("match queries
  directly against cartridge embeddings"). Our head-routing finding is a training-free
  version on a signal nobody looked at; their Fig. 3 (present 78–87 vs absent 27–31) is
  the behavioural evidence the signal exists. The negative half matters equally: the
  all-head average does not discriminate, so head selection *is* the contribution.
  Practical form: calibrate heads once on card↔query pairs, score cards by mass on those
  ~24 heads, run on the agent's streaming thinking tokens.
- **Second cheap relevance signal: Self-Gating** (CDLM). One token with the memory
  active; first-token entropy < ~4.0 means the memory is relevant (balanced acc 79%,
  ~1% overhead, robust over λ∈[3.75,4.25]). A small MLP over [sim, sim-gap, entropy,
  first 128 dims of first-token hidden] ranks candidates nearly free since they share the
  query prefix cache.
- **Per-card cartridges are the worst case for the technique.** CDLM on 159–731-token
  docs loses to plain text RAG at 7B (28.5 vs 47.4 R-1); CAS's 16-slot floor means short
  docs don't actually compress; storage ~141 MiB per 1k slots at 8B (70 MiB per
  512-slot card → 14 GB for 200 cards). Cards are FinQA-like (IDs, structure, references):
  expect low ratios, small k (FinQA *degrades* past k=5), and unreliable ID↔content binding.
  → Use cartridges as the associative index ("which cards"), let the CLI fetch ground
  truth; group cards by subtree/dependency locality into one cartridge per area (CAS
  grouping argument; fewer, more distinct groups route better).
- **Baseline to beat**: embed cards, retrieve top-k, load those (CAS Cartridge-RAG: 3–4×
  fewer tokens at equal accuracy). The interesting claim is catching relevance that
  embedding similarity misses — a card relevant to what the agent is *doing*, sharing no
  vocabulary with what it is *saying*.
- **Amortise the encoder** (Still): one per-layer Perceiver (~50M params for a 4B model;
  a single-block no-self-attention variant scores 0.756 vs 0.708 canonical, so far
  smaller works) trained once against the frozen model, then a card costs one prefill +
  one forward. Output is literally our `Cartridge` tensor shape; un-rotate keys on input,
  re-rotate on output. Kills both the per-card cost and the staleness problem (re-compact
  on every edit). Costs: checkpoint-specific; task-transfer tax (train it on cards, not
  generic prose); single-pass only — recursion is a separate, harder problem.

## 5. Experiments, in order of information per GPU-hour

1. `reposition` on/off × both orders, same 12 questions (minutes).
2. **Two similar corpora at N=2** (e.g. qwen-jax HEAD vs qwen-jax three months ago, or
   two JAX repos). If routing stays clean, similarity is not the driver and N is; if it
   breaks, we have CAS's mechanism and the routing heads are the instrument.
3. Mixed-visibility retrain of A and B with the other as distractor; re-measure the
   head separation. Predicts: sharper routing.
4. Teacher-cache reuse in `distill.py`; decoupled question model (Opus/Sonnet) + 20 Qs
   per call; SCI init. Measure steps-to-heldout.
5. Overlay our per-layer routing map on Diaz's per-layer key-spectrum IQR for the same
   Qwen3 family — should be the same phenomenon two ways.
6. Self-gating entropy as a relevance score on the same 12 questions; compare with head
   mass.
7. Only then: a Still-style compactor trained on starchart cards.

Pointers not read: Doc-to-LoRA (2602.15902, hypernetwork → adapter in one pass); Latent
Context Compilation (2602.21221, same group as CDLM).
