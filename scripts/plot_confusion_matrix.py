"""
Figure 2: confusion_matrix_naive.png

Loads ONLY outputs/multiclass_y.npz (never retrains). Row-normalized confusion
matrix (each row = true class, normalized to sum to 1) so the near-zero apnea /
hypopnea recall is visible despite the huge `normal` majority. Cells annotated
with the raw counts as well.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "outputs", "multiclass_y.npz")
FIG_DIR = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG_DIR, "confusion_matrix_naive.png")


def main():
    d = np.load(NPZ, allow_pickle=True)
    y_true = d["y_true"]
    y_pred = d["y_pred"]
    idx = [int(i) for i in d["label_index"]]
    names = [str(n) for n in d["label_name"]]

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Row-normalize; guard against a class with zero support (none here).
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                        where=row_sums != 0)

    print("Figure 2 — naive multi-class confusion matrix (from multiclass_y.npz)")
    print("  Event-class definition here: NONE — full 4-class matrix, no merging.")
    print("  Row-normalized (row = true class). Diagonal = per-class recall:")
    for i, n in enumerate(names):
        print(f"    {n:<12} recall={cm_norm[i, i]:.4f}  (support={int(row_sums[i,0])})")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Naive multi-class model — row-normalized confusion matrix\n"
                 "(fraction of each true class; raw counts in parentheses)")

    thresh = 0.5
    for i in range(len(names)):
        for j in range(len(names)):
            frac = cm_norm[i, j]
            raw = cm[i, j]
            ax.text(j, i, f"{frac:.2f}\n({raw:,})",
                    ha="center", va="center", fontsize=9,
                    color="white" if frac > thresh else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class (recall on diagonal)")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
