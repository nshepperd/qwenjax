"""Reliability diagrams for the multiple-choice calibration runs.

    python scripts/mcq_calibration_plot.py --out runs/mcq/v2/calibration.png

One panel per cartridge (small multiples). A dot is one stated-confidence value
of the greedy response -- the most likely letter, then its most likely
confidence -- placed at the share of those responses that were right; its area
is how many test questions it covers. Answers (A-D) and abstentions (E, "I
don't know, or it is not in the codebase") are separate series, because the
number attached to an abstention turned out not to mean the same thing: the
base cartridge says "E, 0% confidence" and is right 95% of the time. There are
no connecting lines: most confidence values are never used, and a line across
the gap would draw a relationship that is not in the data. The strip underneath
is the same counts as bars, because a model that says 100% on everything is
one dot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "runs/mcq/v2"

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ANSWER, ABSTAIN = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated all-pairs, light)

PANELS = [
    ("base", "baseline.npz", "Base cartridge", "16k-step attnmse, before any of this"),
    ("warm40", "checkpoints.npz", "After warm start", "40 supervised steps on stated confidence"),
    ("brier60", "checkpoints.npz", "Warm start + 20 RL steps", "reward: 2 if right, minus Brier"),
    ("brier100", "checkpoints.npz", "Warm start + 60 RL steps", "same run, final checkpoint"),
]


def greedy(logp, gold):
    p = np.exp(logp)
    pl = p.sum(-1)
    idx = np.arange(len(gold))
    l = pl.argmax(-1)
    c = (p[idx, l] / pl[idx, l, None]).argmax(-1)  # grid index 0..10
    return l, c, (l == gold).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V2 / "calibration.png"))
    ap.add_argument("--dir", default=str(V2), help="where baseline.npz and checkpoints.npz live")
    ap.add_argument("--abstain-text", default="I don't know, or it is not in the codebase")
    ap.add_argument("--warm-sub", default="40 supervised steps on stated confidence")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
                         "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2})
    fig = plt.figure(figsize=(14, 6.3), facecolor=SURFACE)
    gs = fig.add_gridspec(2, len(PANELS), height_ratios=[4.2, 1.1], hspace=0.10, wspace=0.14,
                          left=0.055, right=0.985, top=0.63, bottom=0.10)
    ticks = np.arange(0, 1.01, 0.2)
    pct = [f"{int(v * 100)}%" for v in ticks]
    cache = {}
    for k, (name, file, title, sub) in enumerate(PANELS):
        z = cache.setdefault(file, np.load(Path(args.dir) / file))
        sub = args.warm_sub if name == "warm40" else sub
        l, c, hit = greedy(z[name], z["gold"])
        n, last = len(hit), z[name].shape[1] - 1
        ax = fig.add_subplot(gs[0, k], facecolor=SURFACE)
        h = fig.add_subplot(gs[1, k], facecolor=SURFACE)
        ax.plot([0, 1], [0, 1], color=TEXT2, lw=1, ls=(0, (4, 4)), zorder=1)
        top = 1
        for sel, color, off, ha in [(l != last, ANSWER, -0.019, "right"), (l == last, ABSTAIN, +0.019, "left")]:
            bins = [b for b in range(11) if (sel & (c == b)).any()]
            x = np.array([b / 10 for b in bins])
            y = np.array([hit[sel & (c == b)].mean() for b in bins])
            cnt = np.array([(sel & (c == b)).sum() for b in bins])
            ax.scatter(x, y, s=np.maximum(36, 1100 * cnt / n), color=color, edgecolor=SURFACE, linewidth=2,
                       zorder=3, clip_on=False)
            h.bar(x + off, cnt, width=0.034, color=color, zorder=2)
            top = max(top, cnt.max() if len(cnt) else 1)
            for xi, ci in zip(x, cnt):
                if ci >= 25:  # label the bars that matter, not every one
                    h.text(xi + off * (0.2 if ha == "right" else 0.4), ci + 0.03 * 330, str(ci), ha=ha, va="bottom",
                           fontsize=8, color=TEXT2)
        ans = l != last
        ece = float(np.mean(np.abs(c[ans] / 10 - np.array([hit[ans & (c == b)].mean() for b in c[ans]])))) if ans.any() else float("nan")
        brier = float(np.mean((c[ans] / 10 - hit[ans]) ** 2))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=0, labelbottom=False, labelleft=(k == 0))
        ax.set_yticklabels(pct)
        ax.grid(color=GRID, lw=0.8)
        for s in ax.spines.values():
            s.set_visible(False)
        if k == 0:
            ax.set_ylabel("share of responses that were right")
            ax.annotate("perfectly calibrated", xy=(0.42, 0.42), xytext=(0.10, 0.70), color=TEXT2, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
            ax.annotate("answers A–D", xy=(1.0, hit[ans & (c == 10)].mean()), xytext=(0.58, 0.97), color=TEXT2,
                        fontsize=9, ha="center", arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
            ab0 = (l == last) & (c == 0)
            if ab0.any():
                ax.annotate("abstentions (E)", xy=(0.0, hit[ab0].mean()), xytext=(0.22, 0.86), color=TEXT2,
                            fontsize=9, arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
        ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=TEXT)
        ax.text(0, 1.225, sub, transform=ax.transAxes, fontsize=9, color=TEXT2)
        ax.text(0, 1.035, f"answers: calibration error {ece:.2f} · Brier {brier:.2f}\n"
                f"abstains on {int((l == last).sum())} of {n}", transform=ax.transAxes, fontsize=8.6, color=TEXT2,
                linespacing=1.35, va="bottom")

        h.set_xlim(-0.05, 1.05)
        h.set_ylim(0, top * 1.35)
        h.set_yticks([])
        h.set_xticks(ticks)
        h.set_xticklabels(pct)
        h.tick_params(length=0)
        for s in ("top", "left", "right"):
            h.spines[s].set_visible(False)
        h.set_xlabel("stated confidence")
        if k == 0:
            h.set_ylabel("questions", labelpad=27)

    fig.text(0.055, 0.95, "Does the cartridge's stated confidence mean anything?", fontsize=14.5,
             fontweight="bold", color=TEXT)
    fig.text(0.055, 0.905, "503 held-out multiple-choice questions about the corpus, 142 of them about things that do "
             "not exist (the right response there is E, abstain).", fontsize=9.8, color=TEXT2)
    fig.text(0.055, 0.873, "Greedy response: a letter and a confidence in steps of 10%. Each dot is one confidence "
             "value; its area is the number of questions.", fontsize=9.8, color=TEXT2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, markeredgecolor=SURFACE) for c in (ANSWER, ABSTAIN)]
    fig.legend(handles, ["answers a letter A–D", f"abstains: E, “{args.abstain_text}”"],
               loc="upper left", bbox_to_anchor=(0.047, 0.862), frameon=False, fontsize=9.5, labelcolor=TEXT2,
               handletextpad=0.3, ncol=2, columnspacing=2.2)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
