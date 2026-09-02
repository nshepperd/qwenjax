# Frozen cartridge corpora

Snapshots of the text that cartridge experiments distill. They are frozen so
that editing the live package does not silently change the corpus, the
initial KV, or the held-out numbers between runs.

- `qwenjax-0d949e6/` — `src/qwen_jax` as of commit 0d949e6 (the commit that
  added `attnmse.py`), extracted with `git archive`. 30 files, ~50k tokens.
  This is the corpus behind every `runs/cart` and `runs/attnmse` artifact
  from 2026-09-01 on; the self-study data in `runs/cart/{train,heldout}.jsonl`
  was generated from an earlier state of the same package.

`scripts/cartridge.py` loads the newest snapshot here by default; pass
`--files`/`--root` to distill something else. To freeze a new snapshot:

    git archive <commit> src/qwen_jax | tar -x -C corpus/qwenjax-<commit>

and point `CORPUS_ROOT` in `scripts/cartridge.py` at it.
