# Evaluation Findings — Sleep Apnea CNN (LOPO)

**Date:** 2026-06-30
**Scope:** Validation of the documented ~91% accuracy figure and an investigation into
whether the model actually detects apnea/hypopnea, plus two loss-function experiments
(class weighting and focal loss) to see if a "step 1 (loss)" fix is sufficient or whether
"step 2 (architecture/representation)" is required.

> All experiments were run from throwaway scripts in a scratchpad. **No repository files
> were modified.** The numbers below come from re-running the repo's own
> `load_and_prepare_data` + `build_cnn_model` inside faithful replicas of
> `scripts/train_model.py`'s LOPO loop.

---

## 1. How accuracy is calculated in the repo

`scripts/train_model.py` uses **Leave-One-Participant-Out (LOPO) cross-validation**:

- One fold per participant (5 participants → 5 folds).
- Each fold trains a **fresh** CNN on the other 4 participants, predicts on the held-out one.
- Predictions are **pooled across all folds**, then scored once (window-weighted).
- Reported accuracy is plain `accuracy_score` (unweighted over windows).
- **No random seed is set in the repo**, so re-runs vary by a few points.

This is methodologically correct LOPO (no patient leaks across train/val within a fold),
but the headline metric (pooled accuracy) is dominated by the majority class.

---

## 2. Dataset composition (the root cause)

8,800 windows total (30 s windows, 50% overlap), 4 classes:

| Class | Windows | Share |
|---|---|---|
| normal | 8,041 | 91.37% |
| hypopnea | 593 | 6.74% |
| apnea | 163 | 1.85% |
| body event | 3 | 0.03% |

**The documented "91.37% accuracy" is exactly the `normal` prevalence (8041/8800).**
Predicting "normal" for every window scores 91.37% — the headline number is the class
imbalance, not skill ("accuracy paradox").

### Per-participant event counts — the structural problem

| Participant | Total | Apnea | Hypopnea |
|---|---|---|---|
| AP01 | 1,822 | 16 | 79 |
| AP02 | 1,769 | 3 | 150 |
| AP03 | 1,696 | **1** | 16 |
| AP04 | 1,932 | **1** | 166 |
| AP05 | 1,581 | **142** | 182 |
| **All** | 8,800 | 163 | 593 |

**Apnea is 87% concentrated in a single participant (AP05: 142/163).** AP03 and AP04 have
*one* apnea window each. Under 5-participant LOPO, no fold ever has both adequate apnea
*training* and a meaningful apnea *test* set. This caps what any model/loss can achieve and
makes per-fold apnea estimates structurally unstable.

---

## 3. Baseline result (validates the original figure)

Seeded baseline, 10 epochs, no class weights:

- Accuracy **0.913** (reproduces the documented ~91%)
- Apnea recall **6/163 = 3.7%**
- Hypopnea recall **0/593 = 0.0%**
- Macro-F1 **0.255**

Confusion matrix (rows = true, cols = pred; order `[apnea, body event, hypopnea, normal]`):

```
[[   6    0    0  157]
 [   0    0    0    3]
 [   3    0    0  590]
 [  16    0    0 8025]]
```

The model is a **near-constant "normal" predictor**. It misses ~96% of apnea and 100% of
hypopnea. Training loss was already **plateaued by epoch 10** in every fold, so the baseline
had genuinely converged to the always-normal minimum (more epochs would not help it).

---

## 4. Experiment: class weighting & focal loss (is "step 1" enough?)

Per-fold **balanced** class weights (recomputed each fold from the training distribution),
plus a focal-loss variant. Longer-training arms used **early stopping on macro-F1**, where
macro-F1 was monitored on a **stratified validation split carved from the training
participants only** — the held-out LOPO participant was never used for stopping (no leakage).

| Config | Accuracy | Apnea recall | Hypopnea recall | Apnea precision | **Macro-F1 (all 4)** |
|---|---|---|---|---|---|
| Baseline, 10ep | 0.913 | 0.037 | 0.000 | 0.240 | **0.255** |
| Class-weighted, 10ep | 0.129 | 0.804 | 0.199 | 0.061 | **0.103** |
| Weighted-CE + ES(macroF1), ≤50ep | 0.546 | 0.791 | 0.083 | 0.053 | **0.222** |
| Focal γ=2 + ES(macroF1), ≤50ep | 0.887 | 0.000 | 0.010 | 0.000 | **0.239** |

**No configuration beats the trivial baseline's macro-F1 of 0.255.**

What each failure looks like:

- **Baseline** fails by ignoring events (recall ≈ 0).
- **Class-weighted (10ep)** inverts the failure: apnea recall jumps to 80% but precision
  collapses to 6% (predicts events everywhere); accuracy craters to 13%, macro-F1 *worsens*
  to 0.103. Its loss was also **not converged at 10 epochs** (wild, non-monotonic) — which
  motivated the longer-training arms.
- **Weighted-CE + early stopping** finds a less extreme false-positive regime (accuracy 0.55)
  but macro-F1 still only 0.222 — below baseline.
- **Focal γ=2** collapsed back to always-normal: loss frozen at 12.0886 every epoch, apnea
  recall 0.000. Its macro-F1 looks "OK" only because reverting to the majority class gives
  high normal-F1. Likely cause: balanced α puts a huge weight (~600) on the 3-window
  `body event` class, destabilizing focal scaling. Possibly rescuable with tuned γ /
  normalized α / dropping `body event` / lower LR — but failed as a drop-in.

### Training curves: thrashing, not learning

Weighted-CE (AP05-holdout): train loss falls 3.7 → ~1.0, but **val macro-F1 never climbs
past ~0.21 and just oscillates**; apnea recall stays pinned at 1.00 (blanket prediction, not
discrimination); hypopnea recall bounces 0.08–0.79 with no trend. Falling train loss with
flat/oscillating val macro-F1 = the model overfitting the false-positive regime, **not
finding generalizable event signal.**

---

## 5. Conclusion — step 1 is not enough; step 2 is necessary

1. **Loss changes relocate the failure, they don't reduce it.** All variants land in a
   macro-F1 band (~0.10–0.26) centered on the do-nothing baseline.
2. **The ceiling is representational, not optimization.** Longer epochs + macro-F1 early
   stopping were tested; the model thrashes instead of converging upward. When train loss
   drops but val macro-F1 won't rise, the features don't separate the classes. Prime suspect:
   **`GlobalAveragePooling1D` averages a 30 s window into one vector, diluting a brief ~10 s
   apnea into the surrounding normal breathing.**
3. **Data structure caps any method.** With apnea 87% in one participant, 5-participant LOPO
   cannot give a stable apnea estimate regardless of architecture — this is an
   evaluation-design / data-quantity problem too.

### Recommended next steps (step 2)

- **Representation:** replace GAP with something that preserves event locality (deeper
  conv + pooling then flatten, attention pooling, or per-timestep labeling).
- **Target/eval redesign:** merge apnea+hypopnea into one "respiratory event" class (positives
  163 → 756, removes the AP05 monopoly), and/or report apnea separately with an explicit
  "n=163, ~1–2 informative participants" caveat. **5 participants is too few for LOPO to give
  a stable apnea estimate.**
- **Then** revisit weighting/focal on top of a representation that can actually see the event
  (drop `body event`, normalize α).

---

## Appendix — reproducing

- Environment: use the venv interpreter directly (`venv/bin/python`); `python3` on PATH is the
  system interpreter and lacks TensorFlow. TF 2.20.0, scikit-learn 1.8.0, Python 3.12.
- Dataset: `Dataset/breathing_dataset.pkl` (git-ignored, ~145 MB). Regenerate with
  `python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"` if absent.
- Experiment scripts lived in a scratchpad (not committed):
  `eval_full.py` (baseline per-class), `eval_weighted.py` (seeded baseline vs class weights
  + per-participant breakdown), `eval_focal.py` (focal + early stopping on macro-F1).
- All experiments seeded (SEED=42) with identical per-fold init across arms, so differences
  reflect the loss/training change, not random initialization.
