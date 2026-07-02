"""
Seeded, reproducible LOPO harness for the sleep-apnea CNN.

Produces two saved prediction sets from *actual* runs (no numbers are copied
from EVALUATION_FINDINGS.md):

  1. Multi-class model (as in scripts/train_model.py): 4 classes
     -> outputs/multiclass_y.npz  (y_true, y_pred, label mapping)

  2. Rebalanced BINARY model (new): drop `body event`, merge apnea+hypopnea into
     one "respiratory event" class vs "normal", SMOTE on TRAIN folds only.
     -> outputs/binary_y.npz      (y_true, y_pred, y_score, label mapping)

Both use one fixed seed and the identical LOPO fold structure (one fold per
participant, fresh model each fold, predictions pooled across folds, scored once).

Run:  ./venv/bin/python scripts/reproducible_harness.py -dataset Dataset/breathing_dataset.pkl
"""

import os
import sys
import pickle
import argparse

# ---------------------------------------------------------------------------
# ONE fixed seed, set before importing / building anything stochastic.
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
# Make TF op-level determinism best-effort (has a runtime cost, worth it here).
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_cnn_model

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

EPOCHS = 10
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data loading (identical construction to scripts/train_model.py)
# ---------------------------------------------------------------------------
def load_and_prepare_data(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    X, y, groups = [], [], []
    for item in data:
        nasal = item["nasal"]
        thoracic = item["thoracic"]
        spo2 = np.repeat(item["spo2"], 8)  # align 4 Hz SpO2 to 32 Hz respiratory
        combined = np.column_stack((nasal, thoracic, spo2))
        X.append(combined)
        y.append(item["label"])
        groups.append(item["participant"])

    return np.array(X), np.array(y), np.array(groups)


# ---------------------------------------------------------------------------
# Experiment 1 — Multi-class LOPO (reproduces the repo's model, seeded + saved)
# ---------------------------------------------------------------------------
def run_multiclass(X, y_raw, groups):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Multi-class LOPO (4 classes)")
    print("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    print("Class mapping:", mapping)

    unique_participants = np.unique(groups)
    all_y_true, all_y_pred = [], []

    for participant in unique_participants:
        print(f"  fold: hold out {participant}")
        train_idx = groups != participant
        val_idx = groups == participant

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_cnn_model((X.shape[1], X.shape[2]), num_classes)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "multiclass_y.npz"),
        y_true=y_true,
        y_pred=y_pred,
        label_index=np.array(sorted(mapping.keys())),
        label_name=np.array([mapping[k] for k in sorted(mapping.keys())]),
    )

    # Metrics — the "event" side here is apnea + hypopnea vs the rest.
    name_to_idx = {v: k for k, v in mapping.items()}
    event_idx = [name_to_idx[n] for n in ("apnea", "hypopnea") if n in name_to_idx]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Accuracy: {acc:.4f}")
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                labels=sorted(mapping.keys()),
                                target_names=[mapping[k] for k in sorted(mapping.keys())],
                                zero_division=0))
    for n in ("apnea", "hypopnea"):
        if n in name_to_idx:
            i = name_to_idx[n]
            rec = recall_score(y_true == i, y_pred == i, zero_division=0)
            f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
            print(f"  {n:<10} recall={rec:.4f}  f1={f1:.4f}")

    print("  Confusion matrix (rows=true, cols=pred), "
          f"order={[mapping[k] for k in sorted(mapping.keys())]}:")
    print(confusion_matrix(y_true, y_pred, labels=sorted(mapping.keys())))
    print(f"\n  saved -> {os.path.join(OUT_DIR, 'multiclass_y.npz')}")
    return event_idx


# ---------------------------------------------------------------------------
# Experiment 2 — Rebalanced BINARY LOPO with SMOTE on train folds only
# ---------------------------------------------------------------------------
def run_binary(X, y_raw, groups):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Rebalanced BINARY LOPO (respiratory event vs normal)")
    print("=" * 70)

    # Drop `body event` windows entirely (only 3 in the whole dataset).
    keep = y_raw != "body event"
    dropped = int((~keep).sum())
    print(f"  dropped `body event` windows: {dropped}")
    Xb = X[keep]
    yr = y_raw[keep]
    gb = groups[keep]

    # Binary target: 1 = respiratory event (apnea OR hypopnea), 0 = normal.
    y = np.where(np.isin(yr, ["apnea", "hypopnea"]), 1, 0).astype(int)
    mapping = {0: "normal", 1: "respiratory_event"}
    print("  Class mapping:", mapping)
    print(f"  positives (event)={int(y.sum())}  negatives (normal)={int((y == 0).sum())}")

    n_features = Xb.shape[1] * Xb.shape[2]  # flatten window for SMOTE
    unique_participants = np.unique(gb)
    all_y_true, all_y_pred, all_y_score = [], [], []

    for participant in unique_participants:
        train_idx = gb != participant
        val_idx = gb == participant

        X_train, y_train = Xb[train_idx], y[train_idx]
        X_val, y_val = Xb[val_idx], y[val_idx]

        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())

        # SMOTE requires >=2 minority samples and k_neighbors < n_minority.
        if n_pos >= 2 and n_neg >= 2:
            k = min(5, n_pos - 1)
            sm = SMOTE(random_state=SEED, k_neighbors=k)
            Xtr_flat = X_train.reshape(len(X_train), n_features)
            Xtr_res, ytr_res = sm.fit_resample(Xtr_flat, y_train)
            Xtr_res = Xtr_res.reshape(-1, Xb.shape[1], Xb.shape[2])
            note = f"SMOTE k={k}: {n_pos}->{int(ytr_res.sum())} pos"
        else:
            Xtr_res, ytr_res = X_train, y_train
            note = f"SMOTE skipped (n_pos={n_pos} too few) — train left imbalanced"

        print(f"  fold: hold out {participant}  "
              f"(train pos={n_pos}, neg={n_neg}; test pos={int(y_val.sum())}, "
              f"neg={int((y_val == 0).sum())}) [{note}]")

        model = build_cnn_model((Xb.shape[1], Xb.shape[2]), 2)
        model.fit(Xtr_res, ytr_res, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        prob = model.predict(X_val, verbose=0)  # softmax over 2 classes
        score = prob[:, 1]                       # P(respiratory event)
        pred = np.argmax(prob, axis=1)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(pred.tolist())
        all_y_score.extend(score.tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_score = np.array(all_y_score)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "binary_y.npz"),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        label_index=np.array([0, 1]),
        label_name=np.array(["normal", "respiratory_event"]),
    )

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)  # PR-AUC = average precision

    print(f"\n  Accuracy:              {acc:.4f}")
    print(f"  Event recall (pos=1):  {rec:.4f}")
    print(f"  Event F1 (pos=1):      {f1:.4f}")
    print(f"  PR-AUC (avg precision): {pr_auc:.4f}")
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=["normal", "respiratory_event"],
                                zero_division=0))
    print("  Confusion matrix (rows=true, cols=pred), order=[normal, event]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"\n  saved -> {os.path.join(OUT_DIR, 'binary_y.npz')}")


# ---------------------------------------------------------------------------
# Leakage-risk audit — surfaced explicitly, not silently ignored.
# ---------------------------------------------------------------------------
def leakage_audit(X, y_raw, groups):
    print("\n" + "=" * 70)
    print("LEAKAGE / VALIDITY AUDIT (read before trusting the numbers)")
    print("=" * 70)
    print(
        "  [OK]   LOPO splits are by participant: no window from the held-out\n"
        "         participant appears in any training fold.\n"
        "  [OK]   SMOTE is fit_resample'd on the TRAIN fold only, AFTER the\n"
        "         held-out participant is removed. Synthetic positives are never\n"
        "         built from test-participant data -> no SMOTE->test leakage.\n"
        "  [OK]   A fresh model is built each fold; no weights carry across folds."
    )
    # Data-structure caveats that limit what the metrics can mean.
    apnea_by_p, hypo_by_p = {}, {}
    for p in np.unique(groups):
        m = groups == p
        apnea_by_p[p] = int((y_raw[m] == "apnea").sum())
        hypo_by_p[p] = int((y_raw[m] == "hypopnea").sum())
    print("\n  [CAVEAT] Event concentration across participants (affects stability,")
    print("           NOT leakage) — per-participant apnea / hypopnea counts:")
    for p in np.unique(groups):
        print(f"           {p}: apnea={apnea_by_p[p]:>4}  hypopnea={hypo_by_p[p]:>4}")
    print(
        "  [CAVEAT] With few participants, a fold whose test set has ~0 positives\n"
        "           yields an unstable/undefined event estimate for that fold.\n"
        "           Metrics are POOLED across folds, which hides this — interpret\n"
        "           the binary event metrics as dataset-wide, not per-participant.\n"
        "  [NOTE]   No preprocessing (normalization/scaling) is fit globally here,\n"
        "           so there is no train/test scaler leakage to worry about; the\n"
        "           model consumes raw stacked windows exactly as train_model.py does."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset file not found:", args.dataset)
        return

    print(f"Seed = {SEED} (numpy, random, tensorflow, PYTHONHASHSEED, SMOTE)")
    X, y_raw, groups = load_and_prepare_data(args.dataset)
    print(f"Loaded {len(X)} windows, shape={X.shape}, participants={list(np.unique(groups))}")

    leakage_audit(X, y_raw, groups)
    run_multiclass(X, y_raw, groups)
    run_binary(X, y_raw, groups)

    print("\nDone. Prediction arrays saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
