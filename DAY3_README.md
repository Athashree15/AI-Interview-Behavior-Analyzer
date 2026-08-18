# Day 3 — Explainability + Ablation Study

## What was built

1. **`training/explain_models.py`** — SHAP feature-importance for the three best-performing models from Day 1/Day 2:
   - DAiSEE `engagement_binary` — XGBoost (F1-macro 0.552, your best)
   - ChaLearn `extraversion` — Random Forest (R² 0.242, your best trait/model combo)
   - ChaLearn `interview` — Random Forest (F1-macro 0.638, your best)

   Produces global SHAP summary plots (which features actually drive each prediction) and a `shap_top_features.json` — this JSON is what the Streamlit dashboard will read from tomorrow to show "why" alongside every score, instead of a bare number.

2. **`training/run_ablation.py`** — a feature-group ablation study. **Important scoping note, and put this in your report explicitly**: the originally-designed ablation (Module 12/28) compared visual/audio/NLP modality contributions — since audio/NLP/fusion are out of the 5-day scope, this is the honest substitute: an ablation *within* the visual pipeline, comparing:
   - Gaze/head-pose features only (5 features)
   - Emotion features only (7 features)
   - Both combined (12 features, what Day 1/2 already used)

   This answers a real, defensible research question — does combining behavioral (gaze/pose) signals with expressive (emotion) signals outperform either alone — without overstating what was actually tested.

## Setup

```bash
pip install -r requirements-training.txt
```

(adds `shap` on top of what you already have)

## Run both

```bash
python training/explain_models.py
python training/run_ablation.py
```

## What to check

- **SHAP plots** (`outputs/plots/shap_summary_*.png`): do the top features make intuitive sense? E.g., for DAiSEE engagement, I'd expect eye-contact ratio and neutral/happy emotion share to rank high; if something like "multiple_faces_detected_frame_count" comes out as the top feature, that's a red flag worth investigating (likely means the model latched onto a data artifact, not real signal).
- **Ablation table** (`outputs/ablation_results.csv`): does `all_features` outperform both single-group variants? If gaze/head-pose alone matches or beats the combined set on some task, that's a legitimate, interesting finding to report (emotion features may be adding noise rather than signal for that specific task) — not a failure.

## What to send back

- Console output of both scripts
- The ablation results table specifically (small enough to paste directly)
- Whether the SHAP top-5 features per model look intuitively sensible to you

Once this is in, Day 3 is locked and we move to **Day 4 — the Streamlit dashboard**, which is the biggest remaining build: wiring the visual pipeline + trained models + SHAP evidence + ablation context into one working app that takes an uploaded video and produces the full analysis.
