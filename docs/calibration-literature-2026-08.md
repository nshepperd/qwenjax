# Calibrated self-knowledge for cartridges — literature synthesis (2026-08-28)

Full report with commentary: https://claude.ai/code/artifact/c4be3b49-a4ff-4cf1-8b54-8420abef0c83
(four parallel Kagi sweeps: verbalized-confidence training, scoring-rule RL, abstention-aware
judging, cartridges/knowledge-injection). This file is the condensed record; the board concept
card is the index.

## Headline findings

1. **The MC scoring rule (binary + Brier on verbalized confidence) is published**: RLCR,
   Damani et al. 2025, arXiv:2507.16806. Theorem: correctness reward + any *bounded* proper
   scoring rule ⇒ optimal policy is accurate AND calibrated. Log score breaks the correctness
   incentive unless the answer is frozen (Rewarding Doubt, arXiv:2503.02623, freezes answers
   and optimizes only confidence tokens). Key baseline result: **plain RLVR destroys verbal
   calibration** (ECE 0.37→0.46 OOD vs 0.03 for RLCR at matched accuracy).
2. **Free-form calibration has a principled published answer**: reader-mediated proper scoring
   (Linguistic Calibration, Band et al. ICML 2024, arXiv:2404.00474, code
   tatsu-lab/linguistic_calibration). Prose z is calibrated iff a reader consuming z makes
   calibrated forecasts on related QA (x,y); reward = log score of the reader's forecast +
   normalization penalty (restores strict propriety) + KL leash. Two stages: *summary
   distillation* SFT (sample M responses, LLM-merges into consensus with frequency-derived
   per-claim confidences — most of the total gain), then PPO. The narrower/broader-response
   problem dissolves: the event space belongs to the QA pairs, not the text; hedging pays
   exactly when it improves the reader's forecast.
3. **Uncertainty-as-neutral judge rewards work**: TruthRL (arXiv:2509.25760, ICML 2026),
   GRPO with +1 correct / 0 abstain / −1 hallucinated. Gains come from knowledge-boundary
   recognition, not blanket conservatism. TIAR (arXiv:2605.25850) re-weights the abstention
   advantage by question difficulty.
4. **Judge caveat**: EMBER (arXiv:2410.20774, NAACL 2025) — every tested LLM judge penalizes
   epistemic weakeners on content-identical responses; >30% of pairwise preferences flip.
   Any hedging-neutral judge must be validated, not assumed. No mainstream long-form
   factuality metric scores hedged claims as neutral by design (VeriScore drops them at
   extraction; FactBench's VERIFY has an Undecidable label but penalizes it in aggregate).
5. **The cartridge niche is open**: full-text scan of Cartridges v3 (arXiv:2506.06266) finds
   zero halluc-/abstain-/refus-/uncertain-. No follow-up trains coverage-boundary awareness.
   Nearest mechanism is CAS distractor mixing (arXiv:2606.04557), framed as accuracy, not
   calibration.

## Cartridge-specific structure

- **Grounded boundary targets are nearly free in self-study**: the teacher has the chunk in
  context, so "does the document discuss X?" gets a grounded answer to distill. Subtlety:
  chunk ≠ corpus — "not in this chunk" ≠ "not in the corpus"; boundary questions need a
  full-context teacher pass or cross-chunk aggregation.
- **Forward vs inverse asymmetry**: "does the cartridge cover X?" is forward and trainable;
  "enumerate what you know" is inverse search, which transformers do at ~0% (Physics of LMs
  3.2, arXiv:2309.14402) unless trained on explicit coverage summaries (self-study's
  structuring/summarization prompts partially provide these). Physics 3.1 (arXiv:2309.14316):
  augmentation diversity is what makes injected knowledge addressable at all.
- **The signal may be latent but unverbalized**: Orgad et al. (arXiv:2410.02707) and
  Inside-Out (arXiv:2503.15299) — internal states encode ~40% more knowledge than generation
  exposes. A cartridge is an activation injection; Lindsey's introspection results
  (arXiv:2601.01828) show models can sometimes detect and name injected content. So probe
  first: if in-corpus vs out-of-corpus is linearly decodable from cartridge-conditioned
  activations, calibration training is plumbing, not knowledge installation.
- **Calibration is a transferable meta-skill** (R-Tuning arXiv:2311.09677; Rewarding Doubt;
  Behaviorally Calibrated RL arXiv:2512.19920 — a 4B trained on math matches frontier
  calibration on SimpleQA). Suggests: train the calibration/abstention behavior once (shared
  prefix or LoRA), let each cartridge supply only content. If RL on cartridge params alone
  plateaus, that factorization is the fallback, not a dead end.
- **Keys route, values store** (Diaz, arXiv:2508.17032): coverage may correlate with
  attention mass on cartridge value slots — second probe target.
- Fine-tuning-injects-hallucination line: Gekhman et al. EMNLP 2024 (arXiv:2405.05904) and
  the 2026 mechanism sequel (arXiv:2604.15574: interference among overlapping semantic
  representations; self-distillation mitigates) — both argue the frozen-weights KL-to-grounded-
  teacher cartridge design is already on the right side; the exposure is concentrated at the
  coverage boundary, which self-study's five seed types never probe.

## Agentic/instrumental line (Emily's RLVR framing)

Uncertainty priced through actions, not statements: an environment with a costed corpus-check
action makes checking optimal iff P(correct) < 1 − c/L — behavioral calibration with the
threshold set by task economics (cf. Kalai et al. arXiv:2509.04664's explicit confidence
targets). RLCR's "RLVR destroys calibration" is about the verbal report channel on single-shot
QA and does not apply; the behavioral failure modes are check-too-rarely / compulsive
re-verification instead. Caveats: trains decision-calibration, not verbalizable calibration
(the gate can stay silent — verbalization is a separate distillation, e.g. from the gate's own
behavior traces); credit assignment is much sparser than a scoring-rule reward. Nearest
literature: RL-for-adaptive-retrieval (Search-R1 and relatives, 2025; Self-RAG, FLARE, SEAKR
gate retrieval on uncertainty but without an explicit cost model). Open and stronger claim:
the knowledge store being something the model was *trained on* (the cartridge), so the gate
must reflect parametric self-knowledge, not generic question difficulty.

## Evaluation protocol (any experiment)

- I-CALM two-stage (arXiv:2604.03904): answer-or-abstain, then force-guess the abstained —
  separates targeted abstention from blanket refusal.
- UNCLE (arXiv:2505.16922): short/long-form questions with shared gold answers.
- LoGU "uncertain accuracy" (arXiv:2410.14309): of the hedged claims, what fraction were
  actually wrong — catches cosmetic hedging.
- Report over-conservatism (prudence/over-conservativeness, arXiv:2312.07000) alongside
  hallucination; each is trivially optimized by sacrificing the other.

## Practical RL gotchas (from the 2026 follow-up literature)

- Bounded rule (Brier), not log, when answer+confidence are jointly optimized (RLCR Thm 1).
- Gradient-mask the confidence reward off the reasoning trace (arXiv:2607.00164) or freeze
  answers (Rewarding Doubt).
- Single combined reward has an accuracy/calibration gradient conflict; decoupling helps
  (DCPO, arXiv:2603.09117).
- Verbalized confidences collapse onto round-number anchors (arXiv:2604.23333).
- Claim-precision rewards hack into terseness or confident tangential recitation (Learning to
  Reason for Factuality, arXiv:2508.05618 — their fix: smoothed precision + log-discounted
  detail + pairwise quality judge).
