"""
Figure 3: before_after.png

Loads ONLY outputs/multiclass_y.npz and outputs/binary_y.npz (never retrains).
Grouped bars comparing the naive multi-class model vs the rebalanced binary model
on three event-detection metrics: event RECALL, event F1, and PR-AUC. Every value
is computed from the saved arrays — nothing is hardcoded.

Apples-to-apples "event" definition (printed at runtime):
  * Multi-class model: the `body event` windows (3) are dropped to match the
    binary model's window set; a window is a POSITIVE event iff its true class is
    apnea OR hypopnea, and it is PREDICTED positive iff the predicted class is
    apnea OR hypopnea. -> identical positive definition to the binary model.
  * Binary model: positive class 1 = respiratory_event (apnea OR hypopnea),
    already merged, with `body event` already dropped.

PR-AUC caveat (also printed): the binary model saved P(event) (`y_score`), so its
PR-AUC is a proper threshold-free average precision. The multi-class run saved only
hard labels (no probabilities), so its "PR-AUC" is average precision over the single
hard operating point — a coarse point estimate, not a threshold-free curve. The bars
are annotated accordingly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(ROOT, "outputs", "multiclass_y.npz")
BIN = os.path.join(ROOT, "outputs", "binary_y.npz")
FIG_DIR = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG_DIR, "before_after.png")


def multiclass_event_arrays():
    """Collapse the 4-class arrays to event(1) vs normal(0), dropping body event."""
    d = np.load(MC, allow_pickle=True)
    y_true = d["y_true"]
    y_pred = d["y_pred"]
    names = [str(n) for n in d["label_name"]]
    name_to_idx = {n: int(i) for i, n in zip(d["label_index"], names)}

    event_ids = [name_to_idx[n] for n in ("apnea", "hypopnea") if n in name_to_idx]
    drop_ids = [name_to_idx[n] for n in ("body event",) if n in name_to_idx]

    keep = ~np.isin(y_true, drop_ids)
    yt = np.isin(y_true[keep], event_ids).astype(int)
    yp = np.isin(y_pred[keep], event_ids).astype(int)
    return yt, yp, len(drop_ids)


def main():
    # --- Multi-class model, collapsed to the same positive definition ---
    mc_true, mc_pred, _ = multiclass_event_arrays()
    mc_recall = recall_score(mc_true, mc_pred, pos_label=1, zero_division=0)
    mc_f1 = f1_score(mc_true, mc_pred, pos_label=1, zero_division=0)
    # No saved probabilities -> AP over the single hard operating point.
    mc_prauc = average_precision_score(mc_true, mc_pred)

    # --- Binary rebalanced model ---
    d = np.load(BIN, allow_pickle=True)
    bin_true = d["y_true"]
    bin_pred = d["y_pred"]
    bin_score = d["y_score"]
    bin_recall = recall_score(bin_true, bin_pred, pos_label=1, zero_division=0)
    bin_f1 = f1_score(bin_true, bin_pred, pos_label=1, zero_division=0)
    bin_prauc = average_precision_score(bin_true, bin_score)  # proper, threshold-free

    prevalence = float(bin_true.mean())  # positive-prevalence PR-AUC baseline

    # Sanity: both models must share the exact same window count & positive count.
    assert len(mc_true) == len(bin_true), "window counts differ between models"
    assert int(mc_true.sum()) == int(bin_true.sum()), "positive counts differ"

    print("Figure 3 — before/after event-detection comparison")
    print("  Positive (event) definition — IDENTICAL for both models:")
    print("    apnea OR hypopnea = positive; normal = negative; body event dropped.")
    print(f"    windows compared: {len(bin_true)}   positives: {int(bin_true.sum())}"
          f"   prevalence: {prevalence:.4f}")
    print("  Multi-class (naive):   recall=%.4f  f1=%.4f  PR-AUC(hard,point)=%.4f"
          % (mc_recall, mc_f1, mc_prauc))
    print("  Binary (rebalanced):   recall=%.4f  f1=%.4f  PR-AUC(threshold-free)=%.4f"
          % (bin_recall, bin_f1, bin_prauc))
    print("  PR-AUC note: multi-class value is average precision at ONE hard operating")
    print("  point (no saved probabilities); binary value is threshold-free from y_score.")
    print(f"  Reference: a no-skill PR-AUC equals the positive prevalence = {prevalence:.4f}.")

    metrics = ["Event recall", "Event F1", "PR-AUC"]
    mc_vals = [mc_recall, mc_f1, mc_prauc]
    bin_vals = [bin_recall, bin_f1, bin_prauc]

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, mc_vals, w, label="Naive multi-class",
                color="#b0b0b0", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, bin_vals, w, label="Rebalanced binary (SMOTE)",
                color="#4c72b0", edgecolor="black", linewidth=0.6)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # No-skill PR-AUC reference line, drawn only across the PR-AUC group.
    pr_i = metrics.index("PR-AUC")
    ax.hlines(prevalence, pr_i - w, pr_i + w, colors="#c44e52",
              linestyles="--", linewidth=1.6, zorder=5)
    ax.text(pr_i + w, prevalence, f"  no-skill PR-AUC\n  = prevalence {prevalence:.3f}",
            va="center", ha="left", fontsize=8, color="#c44e52")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(mc_vals), max(bin_vals), prevalence) * 1.25)
    ax.set_title("Event detection: naive multi-class vs rebalanced binary\n"
                 "(positive = apnea OR hypopnea, identical for both models)")
    ax.legend(loc="upper left")

    # Footnote about the PR-AUC asymmetry so the figure is self-explanatory.
    fig.text(0.5, 0.005,
             "PR-AUC: binary is threshold-free (P(event) saved); multi-class is average "
             "precision at one hard operating point (no probabilities saved).",
             ha="center", va="bottom", fontsize=7.5, color="0.35")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
