# Live Webcam Demo (Demonstration Only)

## What this is (and isn't)

A standalone OpenCV window showing real-time face/head-pose/emotion/gaze
overlay plus a rolling engagement prediction, built entirely from
components you already validated in Phase 2-5. **This is explicitly a
demonstration script, not a deployable/production real-time system** —
say this plainly if asked during your defense. It reuses the same
`FaceLandmarkDetector`, `estimate_head_pose`, `estimate_eye_contact`,
`EmotionClassifier`, and the trained `daisee_engagement_binary_xgboost`
model via the same `build_feature_vector`/`predict_engagement`
functions the Streamlit dashboard uses — nothing here is a separate,
untested reimplementation.

## Setup

Nothing new to install — this uses only packages you already have from
Phase 2 (`mediapipe`, `torch`, `transformers`, `opencv-python`) plus
Day 1's `joblib`/`xgboost`, which you already installed.

## Run it

```bash
python scripts/live_demo.py
```

A window opens showing your webcam feed with overlay text. **Press `q`
in that window to quit** (closing the window with the X button may not
release the camera cleanly — always use `q`).

## Useful flags

```bash
# If you have multiple cameras and the wrong one opens:
python scripts/live_demo.py --camera-index 1

# Slower machine / want it smoother: lower the analysis rate (webcam
# feed itself still displays at full framerate, only the heavy
# analysis is throttled):
python scripts/live_demo.py --process-fps 3

# Faster updates to the engagement estimate:
python scripts/live_demo.py --update-interval 2 --window-seconds 5
```

## What you'll see

- A green box around your detected face
- Live head yaw/pitch angles
- "Looking at camera (approx.)" / "Looking away (approx.)" status
- Top emotion + confidence, updating a few times per second
- **Live engagement estimate** (Low/High + confidence), refreshing every few seconds once enough data has accumulated in the rolling window

## If something breaks

- **Webcam won't open**: try `--camera-index 1` or `2` — index 0 isn't guaranteed to be your primary camera, especially on laptops with a virtual/driver camera also installed.
- **"Trained engagement model not found" warning**: run `training/train_daisee_baseline.py` first if you haven't (Day 1) — the live demo will still show face/pose/emotion overlay without it, just no engagement estimate.
- **Laggy/stuttering feed**: lower `--process-fps` (try 2 or 3).

## For your defense/demo day

Good talking points this naturally supports:
- "The same pipeline code that produced my batch-processed training data is running live here — there's no separate demo-only implementation."
- Point out the **approximate** labeling on eye contact live, reinforcing that you've been careful about this distinction throughout the project, not just in the report.
- The engagement estimate updates on a rolling window, same conceptual design as the temporal-bin aggregation used for the batch DAiSEE/ChaLearn processing — consistent design across offline and live inference.
