# Phase 1 — Dataset Inspection

## What this phase does
Before writing a single line of preprocessing/training code, we verify
that both datasets are structurally what we expect: correct folder
layout, parseable labels, no obvious corruption, and (critically for
DAiSEE) no subject overlap between splits. Nothing here modifies your
dataset — it's read-only inspection.

## Setup (run once)

Open PowerShell or Command Prompt in the project folder:

```bash
cd AI-Interview-Analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements-phase1.txt
```

## Configure paths

Open `configs/config.yaml` and replace the two placeholder paths:

```yaml
datasets:
  daisee:
    root: "PATH/TO/DAiSEE"          # <- change this
  chalearn_fi_v2:
    root: "PATH/TO/FirstImpressionsV2"   # <- change this
```

Use the actual folder that directly contains the dataset's top-level
subfolders (e.g. for DAiSEE, the folder containing `DataSet/` and
`Labels/`; for ChaLearn, the folder containing `train-1`, `train-2`,
..., `train-annotation`, etc.).

## Run the inspection scripts

```bash
python scripts/inspect_daisee.py
python scripts/inspect_chalearn.py
```

## What to send back to me

1. **Full console output** of both scripts (copy-paste is fine, or a
   screenshot if it's long — either works).
2. The two CSV files written to `outputs/`:
   - `daisee_inventory.csv`, `daisee_video_health_sample.csv`
   - `chalearn_inventory.csv`, `chalearn_video_health_sample.csv`
   (just describe row counts/columns if you'd rather not upload the
   full files — the console summary is usually enough for me to work
   from.)

## What I'll be checking for in your output

- **DAiSEE**: subject overlap warnings (should be none/minimal),
  label column names actually matching `Boredom/Engagement/Confusion/Frustration`,
  class distribution (DAiSEE is known to be imbalanced — expect it,
  but I need the real numbers), any corrupt video files.
- **ChaLearn**: which annotation format actually loaded (pickle vs csv vs json),
  whether all five Big-Five traits + `interview` variable were found,
  video/annotation mismatch counts, transcription file presence.

If either script errors out immediately, paste the full traceback —
that's expected to happen at least once (dataset structures vary
between downloads) and is exactly what this phase is for.

## After this

Once I see real output confirming both datasets are usable, we lock
the exact preprocessing spec (frame sampling strategy, face-detection
backend, feature schema) and move to **Phase 2 — Visual pipeline**.
