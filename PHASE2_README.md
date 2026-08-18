# Phase 2 — Visual Pipeline

## What was built

- `src/video/frame_sampler.py` — samples frames at a fixed fps (config-driven), not every native frame.
- `src/video/feature_cache.py` — on-disk caching so a video is never reprocessed twice.
- `src/vision/face_landmarks.py` — MediaPipe Face Mesh wrapper (handles no-face and multi-face cases).
- `src/vision/head_pose.py` — pitch/yaw/roll via solvePnP.
- `src/vision/gaze.py` — approximate eye-contact estimate (head orientation + iris centering). **Read the docstring** — this is explicitly NOT true gaze tracking, and is labeled as an approximation everywhere it surfaces.
- `src/vision/emotion.py` — pretrained ViT emotion classifier (`trpakov/vit-face-expression`), PyTorch-based to match the rest of the stack.
- `src/vision/face_crop.py` — face cropping for the emotion model's input.
- `src/vision/pipeline.py` — orchestrates all of the above per frame, and aggregates into **temporal bins** (default 10-second windows, matching the Module 2 spec) rather than a single video-level average.
- `scripts/run_visual_pipeline_demo.py` — runs the full pipeline on one video, with caching.
- `scripts/extract_chalearn_annotations.py` — kept from before; still useful for reference even though we're skipping the encrypted val/test files (the training pickle already worked).

## Setup

```bash
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-phase2.txt
```

The first run will download the emotion model weights (~350MB) to `models/cache/` — this happens once.

## Run the demo

```bash
python scripts/run_visual_pipeline_demo.py
```

This automatically picks the first video from your Phase-1-generated `outputs/daisee_inventory.csv`. To target a specific file instead:

```bash
python scripts/run_visual_pipeline_demo.py --video "C:/Users/athas/Downloads/DAiSEE_cv/DAiSEE/DataSet/Train/110001/1100011002/1100011002.avi"
```

(use any real path from your `outputs/daisee_inventory.csv`)

## What to check in the output

1. **Face detection rate** — should be high (>90%) for DAiSEE, since subjects are seated facing a webcam. If it's low, something's off with the video codec or lighting and we need to investigate before trusting anything downstream.
2. **Eye-contact ratio / head yaw & pitch stability** — sanity-check these look like plausible numbers (ratios between 0–1, stability std devs in a reasonable degree range, not wildly erratic).
3. **Emotion distribution** — should sum to ~100% and contain recognizable labels (angry/happy/neutral/sad/etc., depending on the model's label set — the actual label names will print in your console output, share them with me).
4. **Runtime** — note how long a single 10-second DAiSEE clip takes to process end-to-end (printed in the log). This tells us whether batch-processing all 9,068 DAiSEE clips is feasible overnight on your machine or needs a smaller subset — send me this number.

## What to send back

- Full console output of `run_visual_pipeline_demo.py` (at least one run)
- The `outputs/visual_pipeline_demo/<video_stem>/result.json` file, or just describe its shape if it's large
- The processing-time-per-clip number specifically

---

## Batch processing the full DAiSEE dataset

Once the demo pipeline checks out on a single clip, use `scripts/batch_process_daisee.py` to run it across all labeled DAiSEE videos (8,986 after dropping the 82 unlabeled ones).

**Always test on a small subset first — do not go straight to the full run:**

```bash
python scripts/batch_process_daisee.py --limit 20
```

Check `outputs/daisee_visual_features_summary.csv` looks right (one row per video, sane-looking values), then run the full thing:

```bash
python scripts/batch_process_daisee.py
```

**This is resumable and crash-safe by design:**
- Every video's result is written to the output CSV *immediately* after processing (flushed to disk), not held in memory — a crash, Ctrl+C, or laptop sleep partway through loses nothing already completed.
- If you stop and re-run the same command, it automatically skips every `clip_id` already present in `daisee_visual_features_summary.csv` — no wasted reprocessing.
- A single corrupt/unreadable video will not kill the run — it's logged to `daisee_visual_features_failures.csv` with the actual error, and processing continues to the next video.
- Progress (avg seconds/video, ETA) prints every 25 videos, computed from actual elapsed time — not a fixed guess.

**Time estimate:** at your measured ~5.7s/video (with model weights already cached), 8,986 videos ≈ **14 hours**. That's a realistic overnight run, but plan accordingly — you can also process one split at a time with `--split train` / `--split validation` / `--split test` if you'd rather break it up.

**Output:**
- `outputs/daisee_visual_features_summary.csv` — one row per video: labels + summary stats (eye-contact ratio, head stability, emotion distribution). This is what Phase 5's baseline models (majority class / logistic regression / random forest) will train on directly.
- `outputs/daisee_visual_features_failures.csv` — any videos that failed, with the actual error message. Review this once the run finishes; a handful of failures out of ~9,000 is expected and fine, a large number means something systemic needs investigating before we trust the dataset.
- `cache/<video_hash>/visual_features.json` — full per-frame and per-temporal-bin detail for every video, which Phase 5's temporal LSTM/GRU/Transformer models will need (they require the actual time sequence, not just summary stats).

## What to send back after the batch run

- Final console summary line (`Processed: X, Failed: Y`)
- Row count and a `head()` of `daisee_visual_features_summary.csv`
- Contents of `daisee_visual_features_failures.csv` if non-empty
- How long the full run actually took

