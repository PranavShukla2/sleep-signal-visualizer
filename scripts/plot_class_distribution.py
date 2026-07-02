"""
Figure 1: class_distribution.png

Loads ONLY outputs/multiclass_y.npz (never retrains). Bar chart of the 4-class
window counts from y_true, annotated with counts + percentages, to show the
imbalance that drives the accuracy paradox.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "outputs", "multiclass_y.npz")
FIG_DIR = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG_DIR, "class_distribution.png")

COLORS = {
    "normal": "#4c72b0",
    "hypopnea": "#dd8452",
    "apnea": "#c44e52",
    "body event": "#8172b3",
}


def main():
    d = np.load(NPZ, allow_pickle=True)
    y_true = d["y_true"]
    names = [str(n) for n in d["label_name"]]        # index-aligned to y_true values
    idx = [int(i) for i in d["label_index"]]

    counts = np.array([int((y_true == i).sum()) for i in idx])
    total = int(counts.sum())
    pct = 100.0 * counts / total

    # Sort descending for readability.
    order = np.argsort(counts)[::-1]
    names = [names[i] for i in order]
    counts = counts[order]
    pct = pct[order]

    print("Figure 1 — class distribution (from multiclass_y.npz y_true)")
    print(f"  Total windows: {total}")
    for n, c, p in zip(names, counts, pct):
        print(f"    {n:<12} {c:>5}  ({p:5.2f}%)")
    print("  Event-class definition here: NONE — this is the raw 4-class ground")
    print("  truth (apnea / hypopnea / body event / normal), no merging.")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts,
                  color=[COLORS.get(n, "#777777") for n in names],
                  edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Number of 30 s windows")
    ax.set_title("Class distribution (ground-truth windows)")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.margins(x=0.05)

    for bar, c, p in zip(bars, counts, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{c:,}\n{p:.2f}%", ha="center", va="bottom", fontsize=10)

    ax.text(0.98, 0.95,
            f"n = {total:,} windows",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
