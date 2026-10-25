# Week 2 — First ML Model

Notebook: [02_First_ML_Model.ipynb](02_First_ML_Model.ipynb)

Regression on scikit-learn's Diabetes dataset (442 patients). Linear Regression vs Random Forest, 80/20 split.

| | Linear Regression | Random Forest |
|---|---|---|
| Test MSE | 2900 | 2998 |
| Test R² | 0.453 | 0.434 |
| Train R² | 0.528 | 0.924 |
| 5-fold CV R² | 0.482 | 0.421 |

The simpler model wins: the Random Forest overfits (train R² 0.92 vs test 0.43).
