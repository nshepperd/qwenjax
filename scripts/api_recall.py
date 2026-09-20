"""Grader-free API-recall benchmark from the frozen corpus AST.

Items are facts a reader of the corpus would know and a grader can check by
string match: the parameters of a function or method, its return annotation,
the module that defines a class or function, a parameter's default, a class's
bases, a field's type. Gold comes from `ast`, so it is exact, and the model is
asked to reply with the bare answer, so grading is a normalised match with no
LLM in the loop. Traps are questions of the same form about entities that do
not exist in the corpus -- a real class with another class's method, a real
function with a borrowed parameter, or a name assembled from corpus vocabulary
that never occurs in the text -- where the only right answer is
NOT IN CODEBASE.

    python scripts/api_recall.py gen   [--out runs/recall/items.jsonl] [--traps 200]
    python scripts/api_recall.py run   dagger=runs/attnmse/dagger.safetensors [rms1=...] \
                                       [--icl] [--init] [--none] [--tag NAME]
    python scripts/api_recall.py score --responses runs/recall/resp-NAME.jsonl

Every item carries where in the corpus its fact lives (`tok_pos`, and
`init_window` for facts inside the cartridge's initialisation), and how many
self-study conversations mention the entity, so recall can be split by what the
cartridge could have retained. `icl` puts a window of the defining file in the
system prompt (open book, the ceiling); `none` is the description alone (what
the model guesses from naming conventions, the floor).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax
import numpy as np

from cartridge import CORPUS_ROOT, DESCRIPTION, init_cartridge, load_corpus, load_model, load_tokenizer
from reader_score import encode_suffix, encode_user, generate, read_jsonl, write_jsonl

OUT_DIR = REPO / "runs/recall"
TRAIN = REPO / "runs/cart/train.jsonl"
SRC = CORPUS_ROOT / "src"

KINDS = ["params", "returns", "module", "default", "bases", "field"]
IDENT = re.compile(r"\*{0,2}[A-Za-z_][A-Za-z0-9_]*")
ABSTAIN = re.compile(r"not\s*in\s*(the\s*)?code\s*base|does\s*not\s*exist|no\s*such|"
                     r"not\s*defined|not\s*found|is\s*not\s*(a\s*)?(part|present)", re.I)

# -----------------------------------------------------------------------------
# questions
# -----------------------------------------------------------------------------

REFUSE = " If {what} does not exist in the codebase, reply exactly NOT IN CODEBASE."
Q = {
    "params": ("In the qwen-jax codebase, what are the parameters of `{qual}`, in order? "
               "Reply with only the comma-separated parameter names, omitting self." + REFUSE),
    "returns": ("In the qwen-jax codebase, what is the return type annotation of `{qual}`? "
                "Reply with only the annotation, exactly as written in the source." + REFUSE),
    "module": ("In the qwen-jax codebase, which module defines `{name}`? Reply with only "
               "the dotted module path, for example qwen_jax.foo." + REFUSE),
    "default": ("In the qwen-jax codebase, what is the default value of the parameter "
                "`{param}` of `{qual}`? Reply with only the value as written in the source."
                " If `{qual}` does not exist in the codebase, or has no parameter `{param}`, "
                "reply exactly NOT IN CODEBASE."),
    "bases": ("In the qwen-jax codebase, what does the class `{qual}` inherit from? Reply "
              "with only the base class name(s) as written in the class statement." + REFUSE),
    "field": ("In the qwen-jax codebase, what is the type annotation of the field `{field}` "
              "of the class `{qual}`? Reply with only the annotation as written in the "
              "source. If `{qual}` does not exist in the codebase, or has no field "
              "`{field}`, reply exactly NOT IN CODEBASE."),
}


def question(item):
    what = f"`{item.get('qual') or item.get('name')}`"
    return Q[item["kind"]].format(what=what, **item)


def forced_question(item):
    """The same question with the refusal option removed."""
    q = question(item)
    return q[: q.index(" If ")] + " Reply with your best guess even if unsure."


# -----------------------------------------------------------------------------
# gen: walk the AST
# -----------------------------------------------------------------------------


def module_of(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def param_names(fn) -> list[str]:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def param_defaults(fn) -> dict[str, str]:
    a = fn.args
    out = {}
    pos = a.posonlyargs + a.args
    for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        out[arg.arg] = ast.unparse(d)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[arg.arg] = ast.unparse(d)
    return out


def is_dunder(name):
    return name.startswith("__") and name.endswith("__")


class Census:
    """Every definition in the corpus with its position, plus vocab for traps."""

    def __init__(self, corpus_text: str, paths: list[Path]):
        self.text = corpus_text
        self.functions = []   # dicts: qual, name, module, node, file, line, offset
        self.classes = []     # dicts: qual, name, module, node, methods{name: fn}, fields{name: ann}
        self.method_names = collections.Counter()
        self.field_names = collections.Counter()
        offset = 0
        for p in paths:
            src = p.read_text()
            header = f"### {p.relative_to(CORPUS_ROOT)}\n"
            body_start = offset + len(header)
            lines = [0]
            for ln in src.splitlines(keepends=True):
                lines.append(lines[-1] + len(ln))
            mod = module_of(p)
            tree = ast.parse(src)
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions.append(dict(qual=f"{mod}.{n.name}", name=n.name, module=mod,
                                               node=n, file=str(p), line=n.lineno,
                                               offset=body_start + lines[n.lineno - 1]))
                elif isinstance(n, ast.ClassDef):
                    methods, fields = {}, {}
                    for m in n.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods[m.name] = m
                        elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                            fields[m.target.id] = ast.unparse(m.annotation)
                    self.classes.append(dict(qual=f"{mod}.{n.name}", name=n.name, module=mod,
                                             node=n, methods=methods, fields=fields,
                                             file=str(p), line=n.lineno,
                                             offset=body_start + lines[n.lineno - 1]))
                    self.method_names.update(k for k in methods if not is_dunder(k))
                    self.field_names.update(list(fields))
            offset = body_start + len(src) + 2  # Corpus.from_files appends "\n\n"
        self.modules = sorted({f["module"] for f in self.functions} | {c["module"] for c in self.classes})
        counts = collections.Counter(d["name"] for d in self.functions + self.classes)
        self.unique = {k for k, v in counts.items() if v == 1}

    def absent(self, s: str) -> bool:
        return s not in self.text


def _words(name: str) -> list[str]:
    if "_" in name or name.islower():
        return [w for w in name.split("_") if w]
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", name)


def _join(words: list[str], camel: bool) -> str:
    return "".join(words) if camel else "_".join(words)


def synth_names(rng, real: list[str], census: Census, n: int, camel: bool) -> list[str]:
    """Names built from the corpus' own vocabulary that never occur in its text."""
    vocab = sorted({w for r in real for w in _words(r)})
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        base = _words(rng.choice(real))
        if len(base) < 2:
            continue
        i = rng.randrange(len(base))
        base[i] = rng.choice(vocab)
        cand = _join(base, camel)
        if cand in seen or not census.absent(cand):
            continue
        seen.add(cand)
        out.append(cand)
    return out


def build_items(census: Census, rng: random.Random, n_traps: int) -> list[dict]:
    items = []

    def add(kind, gold, *, trap=False, host=None, **fields):
        d = dict(kind=kind, gold=gold, trap=trap, **fields)
        if host is not None:
            d.update(file=host["file"], line=host["line"], offset=host["offset"])
        items.append(d)

    funcs = [f for f in census.functions]
    meths = [(c, name, fn) for c in census.classes for name, fn in c["methods"].items()
             if not is_dunder(name) or name in ("__init__", "__call__")]

    # ---- in-corpus items
    for f in funcs:
        fn = f["node"]
        add("params", param_names(fn), qual=f["qual"], host=f, entity=f["name"])
        if fn.returns is not None:
            add("returns", ast.unparse(fn.returns), qual=f["qual"], host=f, entity=f["name"])
        for p, d in param_defaults(fn).items():
            add("default", d, qual=f["qual"], param=p, host=f, entity=f["name"])
        if f["name"] in census.unique:
            add("module", f["module"], name=f["name"], host=f, entity=f["name"])
    for c, name, fn in meths:
        qual = f"{c['qual']}.{name}"
        ent = f"{c['name']}.{name}"
        add("params", param_names(fn), qual=qual, host=c, entity=ent)
        if fn.returns is not None:
            add("returns", ast.unparse(fn.returns), qual=qual, host=c, entity=ent)
        for p, d in param_defaults(fn).items():
            add("default", d, qual=qual, param=p, host=c, entity=ent)
    for c in census.classes:
        if c["name"] in census.unique:
            add("module", c["module"], name=c["name"], host=c, entity=c["name"])
        if c["node"].bases:
            add("bases", [ast.unparse(b) for b in c["node"].bases], qual=c["qual"], host=c,
                entity=c["name"])
        for fname, ann in c["fields"].items():
            add("field", ann, qual=c["qual"], field=fname, host=c, entity=c["name"])

    # ---- traps: same forms, entities that do not exist
    traps = []
    method_pool = sorted(census.method_names)
    field_pool = sorted(census.field_names)
    param_pool = sorted({p for f in funcs for p in param_names(f["node"]) if not p.startswith("*")}
                        | {p for _, _, fn in meths for p in param_names(fn) if not p.startswith("*")})
    fake_funcs = synth_names(rng, [f["name"] for f in funcs], census, n_traps, camel=False)
    fake_classes = synth_names(rng, [c["name"] for c in census.classes], census, n_traps, camel=True)
    fake_methods = synth_names(rng, method_pool, census, n_traps, camel=False)

    def misplaced_method(c):
        # a method some other class has, that this one lacks and is never called on it
        cands = [m for m in method_pool if m not in c["methods"] and census.absent(f"{c['name']}.{m}")]
        return rng.choice(cands) if cands else None

    for _ in range(n_traps * 3):
        form = rng.choice(["misplaced_method", "absent_method", "absent_function", "absent_class",
                           "misplaced_field", "misplaced_param"])
        if form == "misplaced_method":
            c = rng.choice(census.classes)
            m = misplaced_method(c)
            if m is None:
                continue
            kind = rng.choice(["params", "returns"])
            traps.append(dict(kind=kind, gold=None, trap=True, form=form,
                              qual=f"{c['qual']}.{m}", entity=f"{c['name']}.{m}",
                              file=c["file"], line=c["line"], offset=c["offset"]))
        elif form == "absent_method":
            c = rng.choice(census.classes)
            m = rng.choice(fake_methods)
            kind = rng.choice(["params", "returns"])
            traps.append(dict(kind=kind, gold=None, trap=True, form=form,
                              qual=f"{c['qual']}.{m}", entity=f"{c['name']}.{m}",
                              file=c["file"], line=c["line"], offset=c["offset"]))
        elif form == "absent_function":
            fname = rng.choice(fake_funcs)
            kind = rng.choice(["params", "returns", "module", "default"])
            mod = rng.choice(census.modules)
            host = rng.choice([f for f in funcs if f["module"] == mod] or funcs)
            d = dict(kind=kind, gold=None, trap=True, form=form, qual=f"{mod}.{fname}",
                     name=fname, entity=fname, file=host["file"], line=host["line"],
                     offset=host["offset"])
            if kind == "default":
                d["param"] = rng.choice(param_pool)
            traps.append(d)
        elif form == "absent_class":
            cname = rng.choice(fake_classes)
            kind = rng.choice(["module", "bases", "field"])
            host = rng.choice(census.classes)
            d = dict(kind=kind, gold=None, trap=True, form=form, qual=f"{host['module']}.{cname}",
                     name=cname, entity=cname, file=host["file"], line=host["line"],
                     offset=host["offset"])
            if kind == "field":
                d["field"] = rng.choice(field_pool)
            traps.append(d)
        elif form == "misplaced_field":
            c = rng.choice([c for c in census.classes if c["fields"]])
            cands = [f for f in field_pool if f not in c["fields"]]
            traps.append(dict(kind="field", gold=None, trap=True, form=form, qual=c["qual"],
                              field=rng.choice(cands), entity=c["name"], file=c["file"],
                              line=c["line"], offset=c["offset"]))
        elif form == "misplaced_param":
            if rng.random() < 0.5:
                f = rng.choice(funcs)
                have, qual, ent, host = param_names(f["node"]), f["qual"], f["name"], f
            else:
                c, name, fn = rng.choice(meths)
                have, qual, ent, host = param_names(fn), f"{c['qual']}.{name}", f"{c['name']}.{name}", c
            cands = [p for p in param_pool if p not in have]
            traps.append(dict(kind="default", gold=None, trap=True, form=form, qual=qual,
                              param=rng.choice(cands), entity=ent, file=host["file"],
                              line=host["line"], offset=host["offset"]))
    # dedupe, then take n_traps
    seen, uniq = set(), []
    for t in traps:
        key = (t["kind"], t.get("qual"), t.get("name"), t.get("param"), t.get("field"))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    items += uniq[:n_traps]

    for i, it in enumerate(items):
        it["id"] = i
        it["question"] = question(it)
    return items


def annotate_positions(items, tokenizer, corpus, census, p, description):
    """Token position of each fact's definition, and whether the init window holds it."""
    from qwen_jax import chat

    enc = tokenizer(corpus.text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _ in enc["offset_mapping"]]
    head = tokenizer.encode(chat.system_open(f"{description}\n\n"), add_special_tokens=False)
    window = max(p - len(head), 0)
    for it in items:
        tok = bisect.bisect_right(starts, it["offset"]) - 1
        it["tok_pos"] = int(tok)
        it["tok_frac"] = round(tok / len(corpus.ids), 4)
        it["init_window"] = bool(tok < window)


def annotate_train_mentions(items, tokenizer):
    """How many self-study conversations name the entity, and how many were cut from the
    chunk that defines it (so the teacher could see the fact)."""
    if not TRAIN.exists():
        return
    from qwen_jax.selfstudy import load_examples

    exs = load_examples(TRAIN)
    convs = [ex.user + "\n" + ex.assistant for ex in exs]
    chunks = [tokenizer.decode(ex.chunk_ids, skip_special_tokens=False) for ex in exs]
    for it in items:
        short = it["entity"].split(".")[-1]
        it["train_mentions"] = sum(short in c for c in convs)
        if it["trap"]:
            it["train_chunks"] = 0
            continue
        marker = ("class " if it["kind"] in ("bases", "field") or
                  (it["kind"] == "module" and it["name"][:1].isupper()) else "def ")
        it["train_chunks"] = sum((marker + short + "(") in c or (marker + short + ":") in c
                                 for c in chunks)


def cmd_gen(args):
    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, None)
    paths = sorted((SRC / "qwen_jax").rglob("*.py"))
    census = Census(corpus.text, paths)
    # sanity: the offsets we computed must land on the definition lines
    for d in census.functions + census.classes:
        line = corpus.text[d["offset"]:corpus.text.index("\n", d["offset"])]
        assert d["name"] in line, (d["qual"], line)
    rng = random.Random(args.seed)
    items = build_items(census, rng, args.traps)
    annotate_positions(items, tokenizer, corpus, census, args.p, DESCRIPTION)
    annotate_train_mentions(items, tokenizer)
    write_jsonl(items, args.out)
    ins = [i for i in items if not i["trap"]]
    print(f"{len(items)} items: {len(ins)} in-corpus, {len(items) - len(ins)} traps -> {args.out}")
    print("  by kind:", dict(collections.Counter(i["kind"] for i in items)))
    print("  trap forms:", dict(collections.Counter(i["form"] for i in items if i["trap"])))
    print(f"  in init window: {sum(i['init_window'] for i in ins)}/{len(ins)}; "
          f"mentioned in training: {sum(i.get('train_mentions', 0) > 0 for i in ins)}; "
          f"defined in a training chunk: {sum(i.get('train_chunks', 0) > 0 for i in ins)}")


# -----------------------------------------------------------------------------
# grading
# -----------------------------------------------------------------------------


def norm(s: str) -> str:
    s = s.replace("`", "").strip().strip(".").strip()
    s = re.sub(r"\s+", "", s)
    return s.replace("'", '"')


def idents(s: str) -> list[str]:
    return IDENT.findall(s.replace("`", ""))


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def grade(item, response: str) -> dict:
    resp = response.strip()
    abstain = bool(ABSTAIN.search(resp))
    out = dict(abstain=abstain, correct=False, f1=None)
    if item["trap"]:
        out["correct"] = abstain
        return out
    if abstain:
        return out
    kind, gold = item["kind"], item["gold"]
    if kind == "params":
        pred = [t for t in idents(resp.split("\n")[0]) if t not in ("self", "cls")]
        pset, gset = set(pred), set(gold)
        tp = len(pset & gset)
        out["f1"] = 2 * tp / (len(pset) + len(gset)) if (pset or gset) else 1.0
        out["correct"] = pset == gset
    elif kind == "module":
        toks = set(idents(resp))
        mods = {m.split(".")[-1] for m in item.get("_modules", [])}
        want = gold.split(".")[-1]
        others = {m for m in mods if m != want and m != "qwen_jax"}
        out["correct"] = want in toks and not (toks & others)
    elif kind == "bases":
        want = [b.split(".")[-1] for b in gold]
        toks = set(idents(resp))
        out["correct"] = all(w in toks for w in want)
    else:  # returns, default, field: exact text after normalisation
        g, r = norm(gold), norm(resp)
        ok = g in r
        if not ok and (v := _num(g)) is not None:
            ok = any(_num(t) == v for t in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", resp))
        out["correct"] = ok
    return out


# -----------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------


def icl_system(item, tokenizer, max_chars=9000):
    src = Path(item["file"]).read_text()
    lines = [0]
    for ln in src.splitlines(keepends=True):
        lines.append(lines[-1] + len(ln))
    off = lines[item["line"] - 1]
    lo = max(0, off - max_chars // 2)
    hi = min(len(src), lo + max_chars)
    lo = max(0, hi - max_chars)
    rel = Path(item["file"]).relative_to(CORPUS_ROOT)
    return f"{DESCRIPTION}\n\n### {rel}\n{src[lo:hi]}"


def cmd_run(args):
    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, None)
    items = read_jsonl(args.items)
    if args.limit:
        rng = random.Random(0)
        ins = [i for i in items if not i["trap"]]
        traps = [i for i in items if i["trap"]]
        items = rng.sample(ins, min(args.limit, len(ins))) + rng.sample(traps, min(args.limit // 4, len(traps)))
    model = load_model()
    dt = model.cache_dtype()
    conds = []
    for spec in args.cartridges:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, Path(name).stem
        conds.append((name, Cartridge.load(path).prefix(dt)))
    if args.init:
        conds.append(("init", init_cartridge(model, tokenizer, corpus, args.p, DESCRIPTION).prefix(dt)))
    if args.none:
        conds.append(("none", init_cartridge(model, tokenizer, corpus, 0, DESCRIPTION).prefix(dt)))
    if args.icl:
        conds.append(("icl", None))

    if args.forced:
        # No way out: measures what the cartridge holds even when it would rather
        # refuse. Traps have no right answer in this mode, so they are dropped.
        items = [i for i in items if not i["trap"]]
        for i in items:
            i["question"] = forced_question(i)
    modules = sorted({i["gold"] for i in items if i["kind"] == "module" and not i["trap"]})
    out = args.out or OUT_DIR / f"resp-{args.tag}.jsonl"
    t0 = time.time()
    rows = []
    key = jax.random.key(0)
    for name, prefix in conds:
        if name == "icl":
            prompts = [encode_user(tokenizer, icl_system(i, tokenizer), i["question"]) for i in items]
            batch = max(1, args.batch // 4)
        else:
            prompts = [encode_suffix(tokenizer, i["question"]) for i in items]
            batch = args.batch
        texts = generate(model, tokenizer, prompts, prefix=prefix, max_new=args.max_new,
                         temperature=0.0, key=key, batch=batch)
        for it, text in zip(items, texts):
            g = grade({**it, "_modules": modules}, text)
            rows.append({**{k: v for k, v in it.items() if k != "question"},
                         "condition": name, "mode": "forced" if args.forced else "free",
                         "response": text, **g})
        print(f"  {name}: {len(texts)} responses ({time.time() - t0:.0f}s)", flush=True)
        write_jsonl(rows, out)  # after every condition: the icl pass is slow

    print(f"wrote {out}")
    summary = summarise(rows)
    (Path(out).with_name(Path(out).name.replace("resp-", "scores-")).with_suffix(".json")
     ).write_text(json.dumps(summary, indent=1))
    print_summary(summary)


# -----------------------------------------------------------------------------
# score
# -----------------------------------------------------------------------------


def _rate(rows, key="correct"):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def annotate_span_coverage(rows, pointed_path, *, fact_tokens=40):
    """From a pointed self-study file: how many times the span holding each fact was
    pointed at (`span_cover`), and whether it was a held-out span (`span_heldout`).

    A fact is taken to occupy `fact_tokens` tokens from its definition line; any
    pointed span overlapping that window counts.
    """
    exs = read_jsonl(pointed_path)
    counts = collections.Counter(tuple(e["span"]) for e in exs
                                 if e.get("span") and e.get("seed_kind", "").startswith("pointed:"))
    side = Path(pointed_path).with_suffix(".spans.json")
    held = []
    if side.exists():
        meta = json.loads(side.read_text())
        st = meta["span_tokens"]
        held = [(i * st, (i + 1) * st) for i in meta["heldout"]]
    for r in rows:
        lo, hi = r["tok_pos"], r["tok_pos"] + fact_tokens
        r["span_cover"] = int(sum(c for (s, e), c in counts.items() if s < hi and e > lo))
        r["span_heldout"] = bool(any(s <= r["tok_pos"] < e for s, e in held))
    return rows


def summarise(rows):
    out = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        ins = [r for r in rs if not r["trap"]]
        traps = [r for r in rs if r["trap"]]
        s = dict(
            n_in=len(ins), n_trap=len(traps),
            recall=_rate(ins), abstain_in=_rate(ins, "abstain"),
            wrong=float(np.mean([not r["correct"] and not r["abstain"] for r in ins])) if ins else float("nan"),
            trap_abstain=_rate(traps),
            params_f1=float(np.mean([r["f1"] for r in ins if r["f1"] is not None])) if ins else float("nan"),
            by_kind={k: _rate([r for r in ins if r["kind"] == k]) for k in KINDS},
            by_kind_n={k: sum(r["kind"] == k for r in ins) for k in KINDS},
            trap_by_form={},
            init_window=_rate([r for r in ins if r["init_window"]]),
            outside_window=_rate([r for r in ins if not r["init_window"]]),
            n_init_window=sum(r["init_window"] for r in ins),
            mentioned=_rate([r for r in ins if r.get("train_mentions", 0) > 0]),
            unmentioned=_rate([r for r in ins if r.get("train_mentions", 0) == 0]),
            n_mentioned=sum(r.get("train_mentions", 0) > 0 for r in ins),
            in_chunk=_rate([r for r in ins if r.get("train_chunks", 0) > 0]),
            not_in_chunk=_rate([r for r in ins if r.get("train_chunks", 0) == 0]),
            n_in_chunk=sum(r.get("train_chunks", 0) > 0 for r in ins),
        )
        if ins and "span_cover" in ins[0]:
            s.update(
                covered=_rate([r for r in ins if r["span_cover"] > 0 and not r["span_heldout"]]),
                uncovered=_rate([r for r in ins if r["span_cover"] == 0 and not r["span_heldout"]]),
                span_heldout=_rate([r for r in ins if r["span_heldout"]]),
                n_covered=sum(r["span_cover"] > 0 and not r["span_heldout"] for r in ins),
                n_span_heldout=sum(r["span_heldout"] for r in ins),
                by_cover={str(k): _rate([r for r in ins if r["span_cover"] == k])
                          for k in sorted({r["span_cover"] for r in ins})},
            )
        for form in sorted({r["form"] for r in traps}):
            s["trap_by_form"][form] = _rate([r for r in traps if r["form"] == form])
        out[cond] = s
    return out


def print_summary(s):
    conds = list(s)
    w = max(len(c) for c in conds) + 2
    print()
    print(f"{'':{w}} recall  wrong  abst-in  trap-abst  f1(params)  init-win  outside  mentioned  unmention  in-chunk")
    for c in conds:
        x = s[c]
        print(f"{c:{w}} {x['recall']:.3f}  {x['wrong']:.3f}  {x['abstain_in']:.3f}    "
              f"{x['trap_abstain']:.3f}      {x['params_f1']:.3f}       "
              f"{x['init_window']:.3f}     {x['outside_window']:.3f}    {x['mentioned']:.3f}      "
              f"{x['unmentioned']:.3f}      {x['in_chunk']:.3f}")
    print()
    print(f"{'by kind':{w}} " + "  ".join(f"{k:>8}" for k in KINDS))
    for c in conds:
        print(f"{c:{w}} " + "  ".join(f"{s[c]['by_kind'][k]:8.3f}" for k in KINDS))
    print(f"{'n':{w}} " + "  ".join(f"{s[conds[0]]['by_kind_n'][k]:8d}" for k in KINDS))
    forms = sorted({f for c in conds for f in s[c]["trap_by_form"]})
    if forms:
        print()
        print(f"{'trap abstain':{w}} " + "  ".join(f"{f[:14]:>14}" for f in forms))
        for c in conds:
            print(f"{c:{w}} " + "  ".join(f"{s[c]['trap_by_form'].get(f, float('nan')):14.3f}" for f in forms))
    x = s[conds[0]]
    if "covered" in x:
        print()
        print(f"{'pointed spans':{w}}  covered  uncovered  held-out   " +
              "  ".join(f"k={k:>2}" for k in x["by_cover"]))
        for c in conds:
            y = s[c]
            print(f"{c:{w}}  {y['covered']:.3f}    {y['uncovered']:.3f}      {y['span_heldout']:.3f}   " +
                  "  ".join(f"{v:.3f}" for v in y["by_cover"].values()))
        print(f"  n covered={x['n_covered']} held-out={x['n_span_heldout']}")
    print(f"\n  n_in={x['n_in']} n_trap={x['n_trap']} init-window={x['n_init_window']} "
          f"mentioned={x['n_mentioned']} in-chunk={x['n_in_chunk']}")
    print("  recall: in-corpus items answered correctly (abstaining counts as not recalled).")
    print("  wrong: answered and incorrect. trap-abst: traps correctly refused.")


def cmd_score(args):
    rows = read_jsonl(args.responses)
    if args.regrade:
        modules = sorted({r["gold"] for r in rows if r["kind"] == "module" and not r["trap"]})
        for r in rows:
            r.update(grade({**r, "_modules": modules}, r["response"]))
        write_jsonl(rows, args.responses)
    if args.pointed:
        annotate_span_coverage(rows, args.pointed)
    s = summarise(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(s, indent=1))
    print_summary(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--out", default=str(OUT_DIR / "items.jsonl"))
    g.add_argument("--traps", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--p", type=int, default=1024, help="cartridge length, for the init-window split")
    g.set_defaults(fn=cmd_gen)

    r = sub.add_parser("run")
    r.add_argument("cartridges", nargs="*", help="name=path.safetensors")
    r.add_argument("--items", default=str(OUT_DIR / "items.jsonl"))
    r.add_argument("--icl", action="store_true")
    r.add_argument("--init", action="store_true")
    r.add_argument("--none", action="store_true")
    r.add_argument("--p", type=int, default=1024)
    r.add_argument("--tag", default="run")
    r.add_argument("--out")
    r.add_argument("--limit", type=int, help="smoke test: N in-corpus items and N/4 traps")
    r.add_argument("--forced", action="store_true",
                   help="drop the NOT IN CODEBASE option (latent recall; traps skipped)")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--max-new", type=int, default=48)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score")
    s.add_argument("--responses", required=True)
    s.add_argument("--regrade", action="store_true", help="re-run the grader over stored responses")
    s.add_argument("--pointed", help="pointed self-study jsonl: split recall by span coverage")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
