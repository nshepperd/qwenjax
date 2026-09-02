"""Train a cartridge on this codebase and ask it questions.

    python scripts/cartridge.py gen   --n 256 --out runs/cart/train.jsonl
    python scripts/cartridge.py gen   --n 32  --seed 1 --out runs/cart/heldout.jsonl
    python scripts/cartridge.py train --data runs/cart/train.jsonl --heldout runs/cart/heldout.jsonl \
                                      --p 1024 --steps 300 --out runs/cart/qwenjax.safetensors
    python scripts/cartridge.py eval  --cartridge runs/cart/qwenjax.safetensors --heldout runs/cart/heldout.jsonl
    python scripts/cartridge.py ask   --cartridge runs/cart/qwenjax.safetensors "What does KVPrefix.shift do?"

The model is the Q4_K_M GGUF of Qwen3-VL-8B-Instruct, text only, the same
one `bench/quantization.py` measures. The corpus is a frozen snapshot of
`src/qwen_jax` under `corpus/` (see its README), so that editing the live
package does not move the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import jax
import jax.numpy as jnp
import numpy as np


def enable_compilation_cache():
    """Persistent XLA compilation cache, shared by every script that imports
    this module (and the tests). The model is passed to jitted functions as
    an argument rather than baked in, so executables are keyed on shapes and
    code alone and hit across runs; a cartridge step or eval that took a
    minute to compile loads in seconds. The directory is per JAX version:
    different versions sharing one cache upset each other."""
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                               str(Path.home() / ".cache" / "jax-compilation" / jax.__version__))
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


enable_compilation_cache()

REPO = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("QWEN_BENCH_MODELS", "/data/models"))
HF = MODELS / "Qwen3-VL-8B-Instruct"
GGUF = Path(os.environ.get("QWEN_BENCH_GGUF", MODELS / "Qwen3-VL-8B-Instruct-GGUF"))
GGUF_FILE = GGUF / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
# Frozen snapshot of src/qwen_jax; file headers in the corpus text are
# relative to this root, so they read `src/qwen_jax/<module>.py` as before.
CORPUS_ROOT = REPO / "corpus" / "qwenjax-0d949e6"

DESCRIPTION = (
    "Below is a section of the source code of qwen-jax, a JAX/Equinox "
    "implementation of the Qwen3-VL vision-language model. It is part of a "
    "larger corpus containing the whole package."
)


def load_model():
    from qwen_jax.gguf import load_qwen3_gguf

    t = time.time()
    model = load_qwen3_gguf(HF, GGUF_FILE, None)
    print(f"loaded {GGUF_FILE.name} in {time.time() - t:.0f}s", flush=True)
    return model


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF)


def load_corpus(tokenizer, files: list[str] | None, root: str | None = None):
    from qwen_jax.selfstudy import Corpus

    if files:
        paths = [Path(f).resolve() for f in files]
        root = Path(root).resolve() if root else REPO
    else:
        paths = sorted((CORPUS_ROOT / "src" / "qwen_jax").rglob("*.py"))
        root = Path(root).resolve() if root else CORPUS_ROOT
    corpus = Corpus.from_files(tokenizer, paths, root=root)
    print(f"corpus: {len(paths)} files, {len(corpus)} tokens", flush=True)
    return corpus


def cmd_gen(args):
    from qwen_jax.selfstudy import save_examples, self_study

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    model = load_model()
    t = time.time()

    def progress(done, total):
        print(f"  {done}/{total} conversations  ({time.time() - t:.0f}s)", flush=True)

    examples = self_study(
        model, tokenizer, corpus, n=args.n, description=args.description,
        key=jax.random.key(args.seed), seed=args.seed, batch_size=args.batch,
        chunk_tokens=(args.chunk_min, args.chunk_max),
        max_user_tokens=args.max_user, max_assistant_tokens=args.max_assistant,
        temperature=args.temperature, progress=progress,
    )
    save_examples(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out}")
    for ex in examples[:3]:
        print(f"\n[{ex.seed_kind}] USER: {ex.user}\nASSISTANT: {ex.assistant[:400]}")


def init_cartridge(model, tokenizer, corpus, p: int, description: str, *, unit_rms: bool = True):
    """KV of `<|im_start|>system\\n{description}\\n\\n{first tokens of the corpus}`."""
    from qwen_jax import chat
    from qwen_jax.cartridge import Cartridge, CartridgeMeta

    head = tokenizer.encode(chat.system_open(f"{description}\n\n"), add_special_tokens=False)
    ids = np.concatenate([np.asarray(head, np.int32), corpus.head(max(p - len(head), 0))])
    meta = CartridgeMeta(model=GGUF_FILE.name, init_tokens=len(ids), description=description)
    return Cartridge.init_from_tokens(model, ids, meta=meta, unit_rms=unit_rms)


def batches_from(tokenizer, examples, args, *, shuffle: bool, seed: int = 0,
                 description: str | None = None):
    from qwen_jax.distill import make_batch

    if description is None:
        description = args.description
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx) - args.batch + 1, args.batch):
        group = [examples[j] for j in idx[i:i + args.batch]]
        yield make_batch(tokenizer, group, description=description, context=args.context,
                         seq=args.seq, pad_id=tokenizer.pad_token_id)


def wsd_schedule(peak: float, steps: int, *, warmup: int, decay_frac: float):
    """Warmup-stable-decay: linear warmup, constant, then linear decay to 0."""
    import optax

    decay = int(round(steps * decay_frac))
    stable = max(steps - warmup - decay, 0)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak, warmup),
            optax.constant_schedule(peak),
            optax.linear_schedule(peak, 0.0, decay),
        ],
        boundaries=[warmup, warmup + stable],
    )


def cmd_train(args):
    import optax

    from qwen_jax.cartridge import Cartridge
    from qwen_jax.distill import Trainer, evaluate
    from qwen_jax.selfstudy import load_examples

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    train = load_examples(args.data)
    heldout = load_examples(args.heldout) if args.heldout else []
    print(f"{len(train)} train, {len(heldout)} held-out examples")
    model = load_model()

    if args.resume:
        cart = Cartridge.load(args.resume)
    else:
        cart = init_cartridge(model, tokenizer, corpus, args.p, args.description,
                              unit_rms=args.unit_rms)
    print(f"cartridge: p={cart.length}, {cart.num_layers} layers, "
          f"{cart.keys.size * 2 * 4 / 1e6:.0f} MB of float32 parameters")

    held = list(batches_from(tokenizer, heldout, args, shuffle=False)) if heldout else []
    log = []

    def report(step, loss, eval=True):
        row = {"step": step, "loss": loss, "time": time.time() - t0}
        if held and eval:
            row["heldout"] = evaluate(model, cart, held, block=args.block)
        log.append(row)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()), flush=True)

    schedule = wsd_schedule(args.lr, args.steps, warmup=args.warmup, decay_frac=args.decay_frac)
    loss = None
    if args.loss == "attnmse":
        from qwen_jax.attnmse import attnmse_loss

        loss = attnmse_loss
    elif args.loss == "onpolicy":
        import functools

        from qwen_jax.attnmse import attnmse_onpolicy_loss

        loss = functools.partial(attnmse_onpolicy_loss, alpha=args.alpha, target=args.target,
                                 beta=args.beta)
    trainer = Trainer(optax.adam(schedule), block=args.block, loss=loss)
    state = trainer.init(cart)
    t0 = time.time()
    report(0, float("nan"))

    step = 0
    epoch = 0
    recent = []
    while step < args.steps:
        for batch in batches_from(tokenizer, train, args, shuffle=True, seed=epoch):
            if args.loss == "onpolicy" and args.rollout_every:
                if step % args.rollout_every == 0:
                    rollout = cart
                cart, state, loss = trainer.step(model, cart, state, batch, rollout)
            else:
                cart, state, loss = trainer.step(model, cart, state, batch)
            recent.append(float(loss))
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                report(step, float(np.mean(recent)),
                       eval=step % args.eval_every == 0 or step == args.steps)
                recent = []
            if step % args.save_every == 0:
                cart.save(args.out)
            if step >= args.steps:
                break
        epoch += 1

    cart.save(args.out)
    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {args.out}")


def cmd_eval(args):
    from qwen_jax.cartridge import Cartridge
    from qwen_jax.distill import evaluate
    from qwen_jax.selfstudy import load_examples

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    heldout = load_examples(args.heldout)
    model = load_model()
    trained = Cartridge.load(args.cartridge)
    held = list(batches_from(tokenizer, heldout, args, shuffle=False,
                             description=trained.meta.description))
    rows = {
        "trained cartridge": trained,
        "init cartridge (ICL on first p tokens)": init_cartridge(
            model, tokenizer, corpus, trained.length, trained.meta.description),
        "no context (description only)": init_cartridge(
            model, tokenizer, corpus, 0, trained.meta.description),
    }
    print(f"\nheld-out KL(teacher || student), nats/token, {len(held)} batches")
    for name, cart in rows.items():
        print(f"  {name:42s} {evaluate(model, cart, held, block=args.block):.4f}")


def cmd_ask(args):
    from qwen_jax import chat
    from qwen_jax.cartridge import Cartridge
    from qwen_jax.selfstudy import generate_batch

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    model = load_model()
    trained = Cartridge.load(args.cartridge)
    variants = {"trained": trained}
    if not args.trained_only:
        variants["init"] = init_cartridge(model, tokenizer, corpus, trained.length,
                                          trained.meta.description)
        variants["no context"] = init_cartridge(model, tokenizer, corpus, 0,
                                                trained.meta.description)
    suffix = tokenizer.encode(chat.suffix([("user", args.question)], open_assistant=True),
                              add_special_tokens=False)
    im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
    for name, cart in variants.items():
        out = model.generate(
            input_ids=jnp.asarray([suffix], dtype=jnp.int32),
            prefix=cart.prefix(model.cache_dtype()),
            max_new_tokens=args.max_new, key=jax.random.key(args.seed),
            temperature=args.temperature, stop_token_id=im_end, pad_token_id=im_end,
            progress_bar=False,
        )
        gen = out.tokens[0, len(suffix):].tolist()
        if im_end in gen:
            gen = gen[: gen.index(im_end)]
        print(f"\n=== {name} ===\n{tokenizer.decode(gen)}")
    del generate_batch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--files", nargs="*", help="corpus files (default: src/qwen_jax/**/*.py)")
        s.add_argument("--root", help="directory the corpus file headers are relative to")
        s.add_argument("--description", default=DESCRIPTION,
                       help="text prepended to each chunk in the system prompt")

    def shapes(s):
        s.add_argument("--batch", type=int, default=2)
        s.add_argument("--context", type=int, default=2176, help="teacher system-prompt slots")
        s.add_argument("--seq", type=int, default=512, help="conversation slots")
        s.add_argument("--block", type=int, default=128, help="positions per lm_head block")

    g = sub.add_parser("gen")
    common(g)
    g.add_argument("--n", type=int, default=256)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--chunk-min", type=int, default=512)
    g.add_argument("--chunk-max", type=int, default=2048)
    g.add_argument("--max-user", type=int, default=128)
    g.add_argument("--max-assistant", type=int, default=384)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--out", required=True)

    t = sub.add_parser("train")
    common(t)
    shapes(t)
    t.add_argument("--data", required=True)
    t.add_argument("--heldout")
    t.add_argument("--p", type=int, default=1024)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--loss", choices=("kl", "attnmse", "onpolicy"), default="kl",
                   help="training objective: blockwise logit KL, teacher-forced "
                        "attention-output MSE, or the same with the student's own "
                        "stream as the query (held-out eval reports KL either way)")
    t.add_argument("--alpha", type=float, default=1.0,
                   help="onpolicy only: weight of the on-policy term; the rest is "
                        "the teacher-forced term")
    t.add_argument("--target", choices=("dagger", "corrective"), default="dagger",
                   help="onpolicy only: see attnmse_onpolicy_layers")
    t.add_argument("--beta", type=float, default=0.0,
                   help="onpolicy corrective only: fraction of the inherited drift each "
                        "layer is asked to absorb (0 = teacher's output on the "
                        "student's queries)")
    t.add_argument("--rollout-every", type=int, default=0,
                   help="onpolicy only: recompute the student streams from a frozen copy "
                        "of the cartridge every N steps (DAgger iterations); 0 = every "
                        "step, fully online")
    t.add_argument("--raw-params", dest="unit_rms", action="store_false",
                   help="optimise the physical KV instead of the unit-RMS parameterization "
                        "(the pre-2026-09 behaviour, for controls)")
    t.add_argument("--lr", type=float, default=5e-3)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--decay-frac", type=float, default=0.2,
                   help="fraction of steps spent decaying linearly to 0 (WSD)")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=50)
    t.add_argument("--resume")
    t.add_argument("--out", required=True)

    e = sub.add_parser("eval")
    common(e)
    shapes(e)
    e.add_argument("--cartridge", required=True)
    e.add_argument("--heldout", required=True)

    a = sub.add_parser("ask")
    common(a)
    a.add_argument("--cartridge", required=True)
    a.add_argument("--max-new", type=int, default=256)
    a.add_argument("--temperature", type=float, default=0.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--trained-only", action="store_true")
    a.add_argument("question")

    args = p.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "eval": cmd_eval, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()
