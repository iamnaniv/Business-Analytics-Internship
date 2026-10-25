# Screen-recording script (Week 4 demo, ~90 seconds)

1. Terminal: `cd week4-streamlit-app`, `pip install -r requirements.txt`, `streamlit run app.py` (show it starting).
2. Browser opens. Point out the title, the sidebar model info and the disclaimer.
3. "Benign example" is loaded → press **Predict** → green *BENIGN*, probability ≈ 0%.
4. Sidebar → "Malignant example" → **Predict** → red *MALIGNANT*, probability ≈ 100%.
5. Edit one value (e.g. set `worst radius` very high) → Predict again to show the output changes.
6. Type a huge number somewhere to show the "outside the training range" warning.
7. One sentence: "The model is a scaled Logistic Regression from Week 3, loaded once with st.cache_resource."

For Week 3's demo video: scroll the notebook (class balance → CV table → confusion matrices → threshold plot → final table) and say the three points from RESULTS.md.
