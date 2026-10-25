# LinkedIn post drafts (one per week) — replace [GitHub link] before posting

Tip: type `@SkillAudit.ai` and pick the company from the dropdown so the tag actually links.

---
## Week 1 — EDA + Visualisations
Week 1 of my @SkillAudit.ai data science internship: understand the data *before* modelling. 🔍

I explored the Breast Cancer Wisconsin dataset (569 samples, 30 features) with Pandas, Matplotlib and Seaborn and made 5 visualisations. What stood out:

• Only 37% of samples are malignant, so plain accuracy would be misleading later
• Size and shape features (concave points, perimeter, radius) separate the classes best
• Outliers turned out to be real large tumours, so dropping them would have thrown away signal
• 21 feature pairs have correlation above 0.9, so a linear model needs care

Notebook with takeaways: [GitHub link]

#DataScience #EDA #Python #Pandas #DataAnalytics

---
## Week 2 — First ML Model
Week 2 at @SkillAudit.ai: my first predictive model, and my first lesson in why you compare two. 📉

I predicted diabetes progression with Linear Regression vs Random Forest:
• Linear Regression: test R² 0.45, MSE 2900
• Random Forest: test R² 0.43, MSE 2998, but R² of 0.92 on the training data

The "fancier" model memorised noise. The simpler one generalised better, and cross-validation agreed. A big train/test gap is a red flag, not a win.

Code and metrics: [GitHub link]

#MachineLearning #scikitlearn #Python #DataScience

---
## Week 3 — Classification Project
Week 3 at @SkillAudit.ai: a classifier that is *measured properly*. 🎯

Project: detect malignant tumours from 30 image-derived features. An "always benign" model gets 63% accuracy and catches zero cancers, so I focused on precision, recall and F1 and used a confusion matrix.

• Compared Logistic Regression and Random Forest with 5-fold cross-validation
• Chose the decision threshold from training data only (no peeking at the test set)
• Final test result: precision 97.6%, recall 97.6%, F1 0.976. Missed malignant cases dropped from 3 to 1 out of 42

Caveat I'm honest about: 114 test samples means one patient moves recall by ~2 points. This is a learning project, not a diagnostic tool.

Notebook + write-up: [GitHub link]

#MachineLearning #Classification #DataScience #Python

---
## Week 4 — Streamlit Demo App
Week 4 at @SkillAudit.ai: the difference between a notebook and a product. 🚀

I wrapped my classifier in a Streamlit app: enter the measurements, press Predict, get a malignant/benign result with the probability. It loads the saved model once, warns when inputs fall outside the training range, and has one-click real examples.

4 weeks, one project: EDA → models → evaluation → app. Thank you @SkillAudit.ai for the structure and push!

Code, README and demo video: [GitHub link]

#Streamlit #MachineLearning #DataScience #Python #Portfolio
