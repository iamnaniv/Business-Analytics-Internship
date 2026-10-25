"""Re-train the final model from Week 3 and save it for the app.

Run this if you get a scikit-learn version warning when loading model/model.joblib:
    python train_model.py
Same pipeline, split and seed as the Week 3 notebook, so results are identical.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "breast_cancer_wisconsin.csv"
OUT = HERE / "model"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
y = df["malignant"]
X = df.drop(columns=["malignant", "diagnosis"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# threshold: highest cut-off that keeps >= 97% out-of-fold recall (training data only)
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
ok = [t for t in grid if recall_score(y_train, oof >= t) >= 0.97]
threshold = float(max(ok)) if ok else 0.5

model.fit(X_train, y_train)
pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
metrics = {"precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred),
           "f1": f1_score(y_test, pred)}

stats = X_train.describe().T
examples = {
    "benign": X_test[y_test == 0].iloc[0].round(5).to_dict(),
    "malignant": X_test[y_test == 1].iloc[0].round(5).to_dict(),
}
meta = {
    "model": "Logistic Regression (with StandardScaler)",
    "threshold": threshold,
    "features": list(X.columns),
    "feature_stats": {f: {"min": float(stats.loc[f, "min"]), "mean": float(stats.loc[f, "mean"]),
                          "max": float(stats.loc[f, "max"])} for f in X.columns},
    "examples": examples,
    "test_metrics": metrics,
}
joblib.dump(model, OUT / "model.joblib")
json.dump(meta, open(OUT / "model_meta.json", "w"), indent=2)
print(f"saved model (threshold={threshold}) | test precision={metrics['precision']:.3f} "
      f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")
