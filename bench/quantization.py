"""Compare quantized Qwen3-VL checkpoints against a bf16 reference.

The question is whether GGUF k-quants buy anything over bitsandbytes 4-bit --
in accuracy, in speed, or in neither -- on a 16 GB card. Everything runs through
the same `qwen_jax` model code, so the only difference between two rows of the
output is how the weights are stored and multiplied. That is the point of doing
it here rather than against llama.cpp: a llama.cpp-vs-transformers comparison
also measures two different attention implementations, two RoPEs and two
samplers, and cannot separate those from the quantization.

The bf16 reference does not fit in 16 GB, so it runs on the host, which is what
`attention_xla` exists for. It is slow -- minutes per chunk, against seconds on
the GPU -- so `--ref-cache` keeps its outputs between runs.

Accuracy is measured teacher-forced: one forward pass over a fixed text, no
sampling, so nothing depends on a seed.

  perplexity        self-contained, comparable across rows without a reference
  top-1 agreement   fraction of positions whose argmax matches bf16
  KL(bf16 || row)   how far the whole predictive distribution moved, in nats

Perplexity alone can look fine while a model has quietly become a different
model; agreement and KL are what catch that.

Speed is measured on whatever device JAX defaults to, separately for prefill
(one pass over a long prompt) and decode (cached, one token at a time), since
quantized matmuls behave very differently in the two regimes.

Run one variant per process. JAX's async allocator keeps its pool reserved
after a model is dropped, so a second 7 GB model in the same process OOMs on a
16 GB card even though either fits alone; `report` merges the result files.

Usage:
    # accuracy, one variant at a time, then merged into one table
    for v in bnb4 q6k q4km q4kxl; do
        python bench/quantization.py accuracy --variants $v --json out-$v.json
    done
    python bench/quantization.py report out-*.json

    # just the bf16 reference, cached for later runs (slow, host-only)
    python bench/quantization.py reference

    # tokens/sec on the GPU
    python bench/quantization.py speed --variants q6k,q4km,bnb4
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

# `reference` is host-only work, but importing jax still brings up the CUDA
# backend and reserves its pool -- a CPU run sitting on 12 GB of a 16 GB card,
# which then denies it to the GPU runs. The backend is chosen at import, so the
# decision has to be made from argv before jax appears.
if "reference" in sys.argv[1:2]:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

MODELS = Path(os.environ.get("QWEN_BENCH_MODELS", "/data/models"))
HF_BF16 = MODELS / "Qwen3-VL-8B-Instruct"
# Worth overriding to a local disk. GGUFReader mmaps the file and faults it in
# in small pieces, which on a network share runs at about 1 MB/s -- an hour to
# load a 6 GB model that takes seconds from local storage. Safetensors reads
# sequentially and does not have the problem.
GGUF_DIR = Path(os.environ.get("QWEN_BENCH_GGUF", MODELS / "Qwen3-VL-8B-Instruct-GGUF"))

# Not /tmp: that is tmpfs here, and the cached reference is several GB.
CACHE = Path(os.environ.get("QWEN_BENCH_CACHE",
                            Path.home() / ".cache" / "qwen-quant-bench"))

# wikitext-2 test split: the standard perplexity corpus, and small enough that
# the whole thing is one HTTP request.
WIKITEXT_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # "safetensors" | "gguf"
    path: Path
    label: str
    mmproj: Path | None = None
    kwargs: dict = field(default_factory=dict)
    # bf16 weights are 16.4 GB and do not fit a 16 GB card, so that row runs on
    # the host through the XLA attention path.
    host: bool = False


VARIANTS = {
    v.name: v
    for v in [
        Variant("bf16", "safetensors", HF_BF16, "bf16 (unquantized)", host=True),
        Variant("bnb4", "safetensors", MODELS / "Qwen3-VL-8B-Instruct-bnb-4bit",
                "bnb nf4"),
        Variant("bnb4u", "safetensors",
                MODELS / "Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", "bnb nf4 (unsloth)"),
        Variant("q6k", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-Q6_K.gguf", "GGUF Q6_K",
                GGUF_DIR / "mmproj-BF16.gguf"),
        Variant("q4km", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                "GGUF Q4_K_M", GGUF_DIR / "mmproj-BF16.gguf"),
        Variant("q4kxl", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                "GGUF UD-Q4_K_XL", GGUF_DIR / "mmproj-BF16.gguf"),
        # Same files with the fused CuTe kernels turned off, so every matmul
        # dequantizes to a dense array first. The honest row to report wherever
        # the kernels are unavailable, and the baseline they should be measured
        # against where they work.
        Variant("q6k-xla", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-Q6_K.gguf",
                "GGUF Q6_K (no kernel)", GGUF_DIR / "mmproj-BF16.gguf",
                kwargs={"fused": False}),
        Variant("q4km-xla", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                "GGUF Q4_K_M (no kernel)", GGUF_DIR / "mmproj-BF16.gguf",
                kwargs={"fused": False}),
        Variant("q4kxl-xla", "gguf", GGUF_DIR / "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                "GGUF UD-Q4_K_XL (no kernel)", GGUF_DIR / "mmproj-BF16.gguf",
                kwargs={"fused": False}),
    ]
}

QUANTIZED = ["bnb4", "bnb4u", "q6k", "q4km", "q4kxl"]
# bf16 first: it is the floor every quantized row is trying to reach, and
# measuring it against the float32 reference is what separates "quantization
# cost" from "we compute in bfloat16 either way".
ALL_ACCURACY = ["bf16", *QUANTIZED]


def load_variant(v: Variant, dtype=jnp.bfloat16, vision: bool = True):
    """Load one variant.

    `vision=False` skips the mmproj for GGUF rows. The vision tower takes no
    part in a text-only forward pass, and leaving its 1.2 GB out of VRAM is the
    difference between the Q6_K row fitting on a 16 GB card and not -- the
    desktop already holds about 4 GB of it. It changes no measured value, only
    what is resident. The safetensors checkpoints are a single file and keep
    their vision weights either way, so the "weights" column is not comparable
    across the two kinds when this is off.
    """
    if v.kind == "gguf":
        from qwen_jax.gguf import load_qwen3_gguf

        mmproj = v.mmproj if vision else None
        return load_qwen3_gguf(HF_BF16, v.path, mmproj, dtype=dtype, **v.kwargs)
    from qwen_jax.loading import load_qwen3_jax

    return load_qwen3_jax(v.path)


@contextmanager
def on_device(host: bool):
    """Place and run a variant on the host or on the default accelerator.

    `QWEN_JAX_ATTENTION` has to be set explicitly rather than left to its
    default, because the default keys off the *backend*, which stays "gpu"
    even while the weights sit in host memory.
    """
    if not host:
        yield
        return
    previous = os.environ.get("QWEN_JAX_ATTENTION")
    os.environ["QWEN_JAX_ATTENTION"] = "xla"
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            yield
    finally:
        if previous is None:
            del os.environ["QWEN_JAX_ATTENTION"]
        else:
            os.environ["QWEN_JAX_ATTENTION"] = previous


def to_float32(model):
    """Upcast every loaded weight, for the ground-truth reference run."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        model,
    )


def weight_bytes(model) -> int:
    """Device bytes held by the model's parameters."""
    from qwen_jax.param import params_with_paths

    total = 0
    for _, param in params_with_paths(model):
        if not param.is_set():
            continue
        for leaf in jax.tree_util.tree_leaves(param):
            if hasattr(leaf, "nbytes"):
                total += leaf.nbytes
    return total


# =============================================================================
# Corpus
# =============================================================================


def wikitext_text() -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / "wikitext2-test.txt"
    if not raw.exists():
        parquet = CACHE / "wikitext2-test.parquet"
        if not parquet.exists():
            print(f"fetching {WIKITEXT_URL}")
            urllib.request.urlretrieve(WIKITEXT_URL, parquet)
        # Avoid a pyarrow dependency for one column of strings: the parquet is
        # plain-encoded UTF-8, so pull the text out and keep it as text.
        import pyarrow.parquet as pq

        table = pq.read_table(parquet)
        raw.write_text("\n".join(table.column("text").to_pylist()))
    return raw.read_text()


def make_chunks(tokenizer, n_chunks: int, seq_len: int) -> np.ndarray:
    """`n_chunks` disjoint token windows of `seq_len` from wikitext-2."""
    text = wikitext_text()
    ids = tokenizer(text, return_tensors="np")["input_ids"][0]
    need = n_chunks * seq_len
    if len(ids) < need:
        raise ValueError(f"corpus has {len(ids)} tokens, need {need}")
    # Skip the first window: wikitext starts with headers and blank lines, which
    # are unrepresentative and heavily weighted at these chunk counts.
    ids = ids[seq_len : seq_len + need]
    return np.asarray(ids, dtype=np.int32).reshape(n_chunks, seq_len)


# =============================================================================
# Accuracy
# =============================================================================


def chunk_logprobs(model, chunk: np.ndarray) -> np.ndarray:
    """Log-softmax over the vocabulary at every position of one chunk.

    The softmax is taken in float32 whatever the model computes in, so the
    only precision difference between rows is the one being measured.
    """
    input_ids = jnp.asarray(chunk)[None, :]
    attention_mask = jnp.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits is not None
    return np.asarray(jax.nn.log_softmax(out.logits[0].astype(jnp.float32), axis=-1))


def nll(logprobs: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Per-position negative log likelihood of the realised next token."""
    return -logprobs[:-1][np.arange(len(chunk) - 1), chunk[1:]]


def summarize(name: str, label: str, per_chunk: list[dict]) -> dict:
    agg = {"variant": name, "label": label}
    for k in per_chunk[0]:
        agg[k] = float(np.mean([c[k] for c in per_chunk]))
    agg["perplexity"] = float(np.exp(agg["nll"]))
    return agg


def run_accuracy(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_BF16)
    chunks = make_chunks(tokenizer, args.chunks, args.seq_len)
    print(f"corpus: {args.chunks} x {args.seq_len} = {chunks.size} tokens of wikitext-2")

    ref = load_reference(chunks, args)
    rows = []

    if ref is not None and args.reference_row:
        # The reference's own perplexity, straight out of the cached
        # log-probs. It costs nothing to add and it is what makes the other
        # rows readable: without it the table says which quantization is least
        # bad, but not what any of them cost against not quantizing.
        per_chunk = [{"nll": float(nll(np.asarray(ref[i], dtype=np.float32), chunk).mean()),
                      "top1_agreement": 1.0, "kl_nats": 0.0}
                     for i, chunk in enumerate(chunks)]
        rows.append(summarize("f32", "float32 (unquantized)", per_chunk))

    for name in args.variants:
        v = VARIANTS[name]
        print(f"\n=== {v.label}{' [host]' if v.host else ''} ===", flush=True)
        with on_device(v.host):
            t = time.time()
            model = load_variant(v, vision=False)
            print(f"loaded in {time.time() - t:.0f}s, "
                  f"{weight_bytes(model) / 1e9:.2f} GB of weights", flush=True)

            per_chunk = []
            for i, chunk in enumerate(chunks):
                t = time.time()
                lp = chunk_logprobs(model, chunk)
                stats = {"nll": float(nll(lp, chunk).mean())}
                if ref is not None:
                    r = np.asarray(ref[i], dtype=np.float32)
                    stats["top1_agreement"] = float((r.argmax(-1) == lp.argmax(-1)).mean())
                    # KL(reference || variant): how much probability mass the
                    # reference puts where this model does not.
                    stats["kl_nats"] = float((np.exp(r) * (r - lp)).sum(-1).mean())
                per_chunk.append(stats)
                print(f"  chunk {i + 1}/{len(chunks)}  nll={stats['nll']:.4f}"
                      f"  ({time.time() - t:.1f}s)", flush=True)

        rows.append(summarize(name, v.label, per_chunk))
        # Drop the weights before loading the next variant: two 5 GB models do
        # not fit, and JAX only releases device buffers once nothing refers to
        # them.
        del model
        gc.collect()

    report_accuracy(rows, args)


def load_reference(chunks: np.ndarray, args):
    """Reference logprobs for every chunk, computed once and cached."""
    if args.no_reference:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path = reference_path(args)
    if path.exists():
        print(f"reference: {path}")
        # mmap: this is several GB and only one chunk is needed at a time.
        return np.load(path, mmap_mode="r")
    if not args.compute_reference:
        print(f"reference: {path} missing -- reporting perplexity only.\n"
              f"  build it with: python bench/quantization.py reference "
              f"--chunks {args.chunks} --seq-len {args.seq_len}")
        return None
    return compute_reference(chunks, args)


def reference_path(args) -> Path:
    return CACHE / f"ref-f32-{args.chunks}x{args.seq_len}.npy"


def compute_reference(chunks: np.ndarray, args) -> np.ndarray:
    """Ground truth: the unquantized model in float32, on the host.

    float32 rather than bf16 because this is what every other row is measured
    against. bf16 logits carry about three decimal digits, which is coarser
    than the KL differences between good quantizations -- a bf16 reference
    would report its own rounding noise as quantization error. Upcasting costs
    32 GB of host RAM and roughly double the time, and buys a reference that is
    not the thing being measured.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = reference_path(args)

    with on_device(host=True):
        print("loading the unquantized model on the host")
        t = time.time()
        model = to_float32(load_variant(VARIANTS["bf16"]))
        print(f"  loaded in {time.time() - t:.0f}s, "
              f"{weight_bytes(model) / 1e9:.1f} GB in float32")

        # Written straight to disk: at float32 the whole array is several GB.
        out = np.lib.format.open_memmap(
            path, mode="w+",
            dtype=np.float32, shape=(len(chunks), args.seq_len, model.vocab_size))
        for i, chunk in enumerate(chunks):
            t = time.time()
            out[i] = chunk_logprobs(model, chunk)
            print(f"  chunk {i + 1}/{len(chunks)}  ({time.time() - t:.0f}s)", flush=True)
        out.flush()

    del model, out
    gc.collect()
    print(f"wrote {path} ({path.stat().st_size / 1e9:.1f} GB)")
    return np.load(path, mmap_mode="r")


def report_accuracy(rows: list[dict], args) -> None:
    has_ref = "top1_agreement" in rows[0]
    print("\n" + "=" * 78)
    header = f"{'variant':22s} {'ppl':>8s} {'nll':>8s}"
    if has_ref:
        header += f" {'top-1 vs bf16':>14s} {'KL(bf16||x)':>12s}"
    print(header)
    print("-" * 78)
    for r in rows:
        line = f"{r['label']:22s} {r['perplexity']:8.4f} {r['nll']:8.4f}"
        if has_ref:
            line += f" {r['top1_agreement'] * 100:13.2f}% {r['kl_nats']:12.5f}"
        print(line)
    print("=" * 78)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


# =============================================================================
# Speed
# =============================================================================


def run_speed(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_BF16)
    chunk = make_chunks(tokenizer, 1, args.prompt_len)[0]
    prompt = jnp.asarray(chunk)[None, :]

    rows = []
    for name in args.variants:
        v = VARIANTS[name]
        print(f"\n=== {v.label} ===")
        model = load_variant(v)
        gb = weight_bytes(model) / 1e9
        print(f"  weights: {gb:.2f} GB")

        # Prefill: one full pass, no cache. The first call pays compilation.
        def prefill():
            out = model(input_ids=prompt, attention_mask=jnp.ones_like(prompt),
                        last_logit_only=True)
            assert out.last_logits is not None
            return out.last_logits

        prefill().block_until_ready()
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            prefill().block_until_ready()
            times.append(time.perf_counter() - t)
        prefill_s = float(np.median(times))

        # Decode: prompt + max_new_tokens through the cached loop, then subtract
        # the prefill so the number is decode alone.
        def generate():
            return model.generate(
                input_ids=prompt,
                attention_mask=jnp.ones_like(prompt),
                max_new_tokens=args.decode_tokens,
                key=jax.random.key(0),
                temperature=0.0,
                progress_bar=False,
            ).tokens

        generate().block_until_ready()
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            generate().block_until_ready()
            times.append(time.perf_counter() - t)
        total_s = float(np.median(times))
        decode_s = max(total_s - prefill_s, 1e-9)

        rows.append({
            "variant": name,
            "label": v.label,
            "weights_gb": gb,
            "prefill_tok_s": args.prompt_len / prefill_s,
            "decode_tok_s": args.decode_tokens / decode_s,
        })
        print(f"  prefill {args.prompt_len / prefill_s:8.1f} tok/s"
              f"   decode {args.decode_tokens / decode_s:7.2f} tok/s")
        del model, prefill, generate
        gc.collect()

    report_speed(rows, args)


def report_speed(rows: list[dict], args) -> None:
    print("\n" + "=" * 72)
    print(f"{'variant':22s} {'weights':>9s} {'prefill':>13s} {'decode':>13s}")
    print("-" * 72)
    for r in rows:
        print(f"{r['label']:22s} {r['weights_gb']:7.2f} GB "
              f"{r['prefill_tok_s']:9.1f} tok/s {r['decode_tok_s']:9.2f} tok/s")
    print("=" * 72)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


# =============================================================================


def run_report(args) -> None:
    """Merge per-variant result files into one table."""
    rows = []
    for path in args.inputs:
        rows.extend(json.loads(Path(path).read_text()))
    order = ["f32", *ALL_ACCURACY]
    rows.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 99)
    # A row can appear in more than one input file; keep the first of each.
    seen, unique = set(), []
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            unique.append(r)
    rows = unique
    if "prefill_tok_s" in rows[0]:
        report_speed(rows, args)
    else:
        report_accuracy(rows, args)


def parse_variants(s: str) -> list[str]:
    if s == "all":
        return list(ALL_ACCURACY)
    if s == "quantized":
        return list(QUANTIZED)
    names = [n.strip() for n in s.split(",") if n.strip()]
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; choose from {sorted(VARIANTS)}")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("report")
    r.add_argument("inputs", nargs="+", help="result JSON files to merge")
    r.add_argument("--json", help="also write the merged results here")

    for name in ("accuracy", "reference", "speed"):
        s = sub.add_parser(name)
        s.add_argument("--json", help="also write the results here")
        if name != "speed":
            s.add_argument("--chunks", type=int, default=8)
            s.add_argument("--seq-len", type=int, default=1024)
        if name == "accuracy":
            s.add_argument("--variants", type=parse_variants, default="all")
            s.add_argument("--no-reference", action="store_true",
                           help="report perplexity only, skip the bf16 comparison")
            s.add_argument("--compute-reference", action="store_true",
                           help="build the reference now if it is not cached")
            s.add_argument("--reference-row", action="store_true",
                           help="also report the unquantized reference's own perplexity")
        if name == "speed":
            s.add_argument("--variants", type=parse_variants, default="all")
            s.add_argument("--prompt-len", type=int, default=1024)
            s.add_argument("--decode-tokens", type=int, default=64)
            s.add_argument("--repeats", type=int, default=3)

    args = p.parse_args()
    if args.command == "reference":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(HF_BF16)
        compute_reference(make_chunks(tokenizer, args.chunks, args.seq_len), args)
    elif args.command == "accuracy":
        run_accuracy(args)
    elif args.command == "report":
        run_report(args)
    else:
        run_speed(args)


if __name__ == "__main__":
    main()
