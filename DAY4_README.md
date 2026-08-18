# Day 4 — Streamlit Dashboard

## What was built

One cohesive app (`app/streamlit_app.py`), not 7 separate pages, given the 5-day scope — four tabs after analysis:

- **Overview** — Engagement level + confidence, interview-invite impression + confidence, Big-Five trait profile (bar chart with real error bars from Random Forest tree-disagreement — genuine uncertainty, not decorative).
- **Visual Trends** — eye-contact ratio, head pose, and emotion distribution over time, straight from the pipeline's temporal bins.
- **Evidence & Explainability** — the Day 3 SHAP top-features per model, framed honestly as *"what typically drives this model's predictions"* (global/training-set level), not a live per-video attribution — plus the actual measured feature values for this video.
- **Recommendations** — rule-based, generated only from this video's actual measured metrics (see `app/recommendations.py` for the exact thresholds and why they're not overclaimed as calibrated/validated).

**Supporting files:**
- `app/model_loader.py` — cached loading of all 7 trained models + Day 3 SHAP JSON + DAiSEE-train-split feature medians (fallback for any feature a given video fails to produce).
- `app/inference.py` — builds the 12-feature vector from a pipeline result and runs all models. Kept separate from the UI file (Module 24: training/inference/UI/config must stay separated).
- `app/recommendations.py` — the grounded recommendation rules.

**Error handling built in:** unreadable/corrupt video → friendly message, not a traceback. Low face-detection rate → warning banner, not silent bad results. Missing model files → sidebar warning naming exactly which predictions are unavailable. Missing features for a given video → disclosed imputation, not silent substitution.

**Caching:** uploaded videos are hashed by content (not by temp file path, since that changes every upload) — re-analyzing the same video in the same session reuses the cached pipeline result instead of reprocessing.

## Setup

```bash
pip install -r requirements-app.txt
```

## Run it

```bash
streamlit run app/streamlit_app.py
```

This opens in your browser automatically (usually `http://localhost:8501`).

## What to test

1. **Upload a real DAiSEE or ChaLearn clip** (pick one from your inventory CSVs) and click Analyze. Check all four tabs render without errors.
2. **Try a very short or unusual video** if you have one, to see the low-frame-count / low-face-detection warnings actually trigger.
3. **Check the Big-Five bar chart** — scores should be roughly in the 0–1 range matching what you saw in Day 2's training output, not wildly different.
4. **Confirm the SHAP evidence tab** shows the same top features you saw in Day 3's console output.

## What to send back

- A screenshot or description of each of the 4 tabs with a real video loaded
- Any error messages (paste the full text — Streamlit errors are usually more useful than a screenshot for these)
- Whether the numbers you see match what you'd expect from Day 1–3's results

Once this is confirmed working, **Day 5** is README + Responsible AI section + limitations write-up + final packaging — the last step before submission.
