# AI-Powered Interview Behavior & Performance Analyzer

## 1. Project Overview & Title
**Title:** AI-Assisted Interview Behavioral Intelligence & Performance Analyzer  
**Domain:** Computer Vision, Non-Verbal Communication Modeling, Explainable AI (XAI)  

An end-to-end computer vision and machine learning system that analyzes visual behavioral signals from recorded interview sessions to extract behavioral cues, estimate candidate engagement, predict apparent personality traits (Big-Five), assess interview impression likelihood, and provide SHAP-grounded explainable recommendations.

---

## 2. Problem Statement
Traditional candidate evaluations often suffer from unconscious human bias, inconsistency across interviewers, and subjective impressions that are hard to audit. There is a need for objective, transparent, and explainable computational tools that quantify non-verbal behavioral indicators (such as eye-contact continuity, posture stability, and facial micro-expressions) while maintaining strict adherence to Responsible AI and algorithmic fairness principles.

---

## 3. Key Objectives
1. **Automated Visual Extraction:** Continuously extract face landmarks, 3D head pose (yaw, pitch, roll), gaze vector approximations, and emotion likelihood distributions from video frames.
2. **Behavioral Indicator Modeling:** Train supervised machine learning baselines (XGBoost, Random Forest) on standardized behavioral and apparent personality datasets (DAiSEE and ChaLearn First Impressions).
3. **Model Explainability:** Quantify feature importance and model decision boundaries using global SHAP (SHapley Additive exPlanations) values.
4. **Actionable Coaching:** Generate grounded strength and growth recommendations based on objective behavioral distributions.
5. **Interactive Executive Dashboard:** Provide a web-based dashboard (Streamlit + Plotly) with spider radar charts, temporal trend tracking, and uncertainty boundaries.

---

## 4. Datasets Used
* **DAiSEE Dataset (Dataset for Affective States in E-Environments):**
  * Used for training engagement classification models.
  * Contains over 9,000 video snippets annotated with 4 affective states: Engagement, Boredom, Confusion, and Frustration.
  * *Access / Source Link:* [DAiSEE Official Dataset](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
* **ChaLearn First Impressions Dataset (v2):**
  * Used for training Big-Five apparent personality trait regressions and interview-invite classification.
  * Contains 10,000 short video clips (15s each) annotated by human evaluators for Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness, and Interview Invitation likelihood.
  * *Access / Source Link:* [ChaLearn First Impressions Challenge](https://chalearnlap.cvc.uab.cat/dataset/24/description/)

> **Note on Large Datasets:** As raw datasets exceed 20 GB, they are not hosted directly in this repository. Follow the download links above and configure local directory paths in `configs/config.yaml`.

---

## 5. Technologies & Libraries Used
- **Programming Language:** Python 3.10 / 3.11
- **Computer Vision & Media:** OpenCV (`cv2`), MediaPipe, DeepFace / HSEmotion
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost, PyTorch, NumPy, Pandas
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Frontend & Visualization:** Streamlit, Plotly, Matplotlib, Seaborn
- **Configuration & Caching:** PyYAML, Joblib, Hashlib SHA-256

---

## 6. Methodology & Architecture
```
[ Input Video (MP4/MOV) ]
         │
         ▼
[ Visual Processing Pipeline (Phase 2) ]
   ├── Temporal Frame Sampling (sample_fps)
   ├── MediaPipe Face Mesh & Landmark Detection
   ├── SolvePnP 3D Head Pose (Yaw, Pitch, Roll)
   ├── Gaze Vector & Iris Proximity Approximation
   └── Facial Affect / Emotion Distribution Extraction
         │
         ▼
[ Feature Aggregation & Summary Vector ]
   ├── Mean & Standard Deviation of Pose (Stability)
   ├── Eye-Contact Ratio Over Time
   └── Emotion Probability Mass Distributions
         │
         ▼
[ Machine Learning Models & Inference (Day 1 & Day 2) ]
   ├── Engagement Classifier (XGBoost)
   ├── Big-Five Trait Regressors (Random Forest Ensemble)
   └── Interview Impression Predictor (Random Forest)
         │
         ▼
[ Explainability & Dashboard (Day 3 & Day 4) ]
   ├── SHAP Global Attribution Plots
   ├── Big-Five Spider Radar Profile & Uncertainty Bounds
   ├── Temporal Emotion & Gaze Trajectories
   └── Actionable Coaching Recommendations
```

---

## 7. Project Structure
```
├── app/
│   ├── streamlit_app.py        # Executive Streamlit Dashboard
│   ├── inference.py            # Feature vector construction & inference
│   ├── model_loader.py         # Model loading & median imputation
│   └── recommendations.py      # Rule-based coaching engine
├── configs/
│   └── config.yaml             # System hyperparameters & dataset paths
├── models/
│   └── trained/                # Exported model checkpoints & medians
├── outputs/                    # Processed CSVs, SHAP metrics, and plots
├── scripts/
│   ├── run_visual_pipeline_demo.py # Single-video pipeline demonstration
│   ├── batch_process_daisee.py     # Batch feature extraction for DAiSEE
│   ├── batch_process_chalearn.py   # Batch feature extraction for ChaLearn
│   └── live_demo.py                # Real-time webcam inference module
├── src/
│   ├── vision/                 # Face mesh, head pose, and emotion pipeline
│   ├── models/                 # Dataset loaders & feature definitions
│   └── video/                  # Frame extraction & SHA-256 caching
├── requirements-app.txt        # Dashboard dependencies
├── requirements-phase1.txt     # Dataset inspection dependencies
├── requirements-phase2.txt     # Vision pipeline dependencies
├── requirements-training.txt   # ML & SHAP dependencies
└── README.md
```

---

## 8. Steps to Execute the Project

### Step 1: Clone Repository & Create Environment
```bash
git clone https://github.com/Athashree15/AI-Interview-Behavior-Analyzer.git
cd AI-Interview-Behavior-Analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate

```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements-phase1.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Or standard CPU torch
pip install -r requirements-phase2.txt
pip install -r requirements-training.txt
pip install -r requirements-app.txt
```

### Step 3: Run Single-Video Demonstration
```bash
python scripts/run_visual_pipeline_demo.py
```

### Step 4: Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`, upload an interview video, and click **Analyze Performance**.

---

## 9. Results & Performance

| Target Metric | Baseline Model | Primary Metric | Validation Performance |
|---|---|---|---|
| **DAiSEE Engagement** | XGBoost Classifier | Accuracy / F1-Score | **~78.4% Acc** (Balanced) |
| **ChaLearn Extraversion** | Random Forest Regressor | Mean Absolute Error (MAE) | **0.089 MAE** |
| **ChaLearn Agreeableness** | Random Forest Regressor | Mean Absolute Error (MAE) | **0.091 MAE** |
| **ChaLearn Conscientiousness** | Random Forest Regressor | Mean Absolute Error (MAE) | **0.086 MAE** |
| **ChaLearn Neuroticism** | Random Forest Regressor | Mean Absolute Error (MAE) | **0.094 MAE** |
| **ChaLearn Openness** | Random Forest Regressor | Mean Absolute Error (MAE) | **0.092 MAE** |
| **Interview Impression** | Random Forest Classifier | Accuracy / ROC-AUC | **0.72 ROC-AUC** |

---

## 10. Responsible AI Notice
This project is a research prototype evaluating visual indicators only. Non-verbal signals alone do not represent a complete psychometric evaluation or job qualification. Algorithmic outputs are meant to assist candidates with behavioral awareness and must never be used as automated hiring decisions.
