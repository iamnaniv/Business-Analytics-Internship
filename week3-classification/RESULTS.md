# Week 3 — Results write-up

Full code, outputs and charts: [03_Classification_Project.ipynb](03_Classification_Project.ipynb)


**Setup.** 569 samples, 30 features, stratified 80/20 split (455 train / 114 test; 42 malignant in test). Logistic Regression (with scaling) vs Random Forest.

| Cross-validated (train, 5-fold) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.974 | 0.977 | 0.953 | **0.964** | 0.996 |
| Random Forest | 0.960 | 0.958 | 0.935 | 0.946 | 0.988 |

Logistic Regression was selected: it wins on every cross-validated metric, is far smaller and faster, and its coefficients are explainable. (On the single small test split the two are within one patient of each other, which is exactly why cross-validation decides.)

**Final model on the held-out test set**

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.5 (default) | 71 | 1 | 3 | 39 | 0.965 | 0.975 | 0.929 | 0.951 |
| **0.3 (chosen)** | 71 | 1 | **1** | 41 | **0.982** | 0.976 | **0.976** | **0.976** |

- **Accuracy is not the headline.** An "always benign" model scores 63% accuracy while catching zero cancers; that is why precision, recall and F1 are reported for the malignant class.
- **Threshold choice.** The threshold of 0.3 was picked from out-of-fold predictions on the *training* data (highest threshold keeping recall ≥ 97%), then applied once to the test set. It cut missed malignant cases from 3 to 1 without adding any false alarms.
- **Reading the errors.** At 0.3 the model misses 1 of 42 malignant samples (false negative) and wrongly flags 1 of 72 benign ones (false positive).
- **Drivers.** Positive coefficients (towards malignant) are led by `worst_texture`, `radius_error`, `worst_symmetry`, `mean_concave_points` and `worst_concavity`; the largest negative ones are `compactness_error` and `fractal_dimension_error`. Because features are correlated (Week 1), individual coefficients should be read as a group effect, not as isolated causes.

**Limitations.** Only 114 test samples: one patient shifts recall by ~2.4 points, so these numbers carry real uncertainty (a confidence interval would be wide). The data comes from one source; performance on other hospitals' images is unknown. This is a learning project, **not a diagnostic tool**.

**Next steps.** Repeated cross-validation for tighter estimates, calibration of probabilities, trying gradient boosting, and testing on an external dataset.
