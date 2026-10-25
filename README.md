# SkillAudit.ai Data Science Internship — 4-week project

An end-to-end machine-learning project built week by week: understand the data → first models → a properly evaluated classifier → a usable app.

| Week | Task | Folder |
|---|---|---|
| 1 | EDA + 5 visualisations (Breast Cancer Wisconsin, UCI) | [week1-eda](week1-eda) |
| 2 | First ML model: compare 2 algorithms (regression: MSE / R²) | [week2-first-model](week2-first-model) |
| 3 | Classification project: confusion matrix, precision / recall / F1 | [week3-classification](week3-classification) |
| 4 | Streamlit demo app that runs locally | [week4-streamlit-app](week4-streamlit-app) |

**Headline result:** a scaled Logistic Regression detects malignant tumours with 97.6% precision and 97.6% recall (F1 0.976) on a held-out test set, after choosing the decision threshold from cross-validated training data only.

```
data/                      breast_cancer_wisconsin.csv (UCI, exported from scikit-learn)
week1-eda/                 notebook + README
week2-first-model/         notebook + README
week3-classification/      notebook + RESULTS.md
week4-streamlit-app/       app.py, predict.py, train_model.py, model/, README
```

Notebooks are committed with their outputs, so they render fully on GitHub. To re-run: `pip install pandas scikit-learn matplotlib seaborn jupyter`, then `jupyter notebook`.

> Educational project — not a medical diagnosis tool.
