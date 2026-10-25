# Week 4 — Streamlit Demo App

A small web app that wraps the Week 3 classifier: enter the 30 measurements of a breast-mass sample, press **Predict**, and get *malignant / benign* plus the probability.

## How to run

```bash
# 1. clone the repo and go to the app folder
cd skillaudit-internship/week4-streamlit-app

# 2. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. start the app (opens http://localhost:8501)
streamlit run app.py
```

## Using the app
- The sidebar loads a real **benign** or **malignant** test sample, or the dataset average.
- Edit any of the 30 values (grouped into mean / standard error / worst) and press **Predict**.
- Values outside the training range trigger a warning that the prediction may be unreliable.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI; loads the model once with `st.cache_resource` |
| `predict.py` | Prediction logic (separate from the UI so it is easy to test) |
| `train_model.py` | Re-trains the Week 3 model (same split/seed) and saves `model/` |
| `model/model.joblib`, `model/model_meta.json` | Saved pipeline (scaler + logistic regression) and threshold, feature ranges, examples |

**If you see a scikit-learn version warning** when the app starts, run `python train_model.py` once to regenerate the model for your installed version.

## Model
Logistic Regression with standardisation; decision threshold **0.30** (chosen from cross-validated training predictions so that malignant cases are rarely missed). Held-out test: precision 0.976, recall 0.976, F1 0.976.

> Educational project — not a medical diagnosis tool.
