"""Streamlit demo: predict whether a breast-mass sample is malignant or benign.

Run:  streamlit run app.py
"""
import streamlit as st

from predict import load_artifacts, predict

st.set_page_config(page_title="Tumour Classifier Demo", page_icon="🔬", layout="wide")


@st.cache_resource  # load the saved model once, not on every interaction
def get_model():
    return load_artifacts()


model, meta = get_model()
features = meta["features"]
stats = meta["feature_stats"]

st.title("🔬 Breast Mass Classifier")
st.caption("Week 4 demo · Logistic Regression trained on the Breast Cancer Wisconsin (Diagnostic) dataset. "
           "Educational project — **not a medical diagnosis tool**.")

# --- sidebar: model info + example loaders ---------------------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {meta['model']}")
    st.write(f"**Decision threshold:** {meta['threshold']} (flag as malignant when probability ≥ this)")
    m = meta["test_metrics"]
    st.write(f"**Held-out test:** precision {m['precision']:.2f} · recall {m['recall']:.2f} · F1 {m['f1']:.2f}")
    st.divider()
    st.header("Try an example")
    choice = st.radio("Load a real test sample", ["Benign example", "Malignant example", "Dataset average"])

if choice == "Benign example":
    defaults = meta["examples"]["benign"]
elif choice == "Malignant example":
    defaults = meta["examples"]["malignant"]
else:
    defaults = {f: stats[f]["mean"] for f in features}

# --- inputs: 30 features in three groups -------------------------------------------------------
st.subheader("Enter feature values")
groups = {
    "Mean values": [f for f in features if f.startswith("mean ")],
    "Standard error": [f for f in features if f.endswith(" error")],
    "Worst (largest) values": [f for f in features if f.startswith("worst ")],
}
values = {}
cols = st.columns(3)
for col, (title, feats) in zip(cols, groups.items()):
    with col:
        st.markdown(f"**{title}**")
        for f in feats:
            lo, hi = stats[f]["min"], stats[f]["max"]
            step = max((hi - lo) / 200, 1e-4)
            # key includes the example choice so the widgets reset when a new example is loaded
            values[f] = st.number_input(f, min_value=0.0, value=float(defaults[f]), step=float(step),
                                        format="%.4f", key=f"{choice}-{f}",
                                        help=f"Training range: {lo:.4g} – {hi:.4g}")

# --- prediction -------------------------------------------------------------------------------
st.divider()
if st.button("Predict", type="primary"):
    proba, is_malignant = predict(model, meta, values)
    left, right = st.columns([1, 2])
    with left:
        if is_malignant:
            st.error("Prediction: **MALIGNANT**")
        else:
            st.success("Prediction: **BENIGN**")
        st.metric("Probability of malignancy", f"{proba:.1%}")
    with right:
        st.progress(min(proba, 1.0))
        st.caption(f"Threshold for flagging malignant: {meta['threshold']:.0%}. "
                   "The threshold is set low on purpose so that malignant cases are rarely missed.")
        out_of_range = [f for f in features if not stats[f]["min"] <= values[f] <= stats[f]["max"]]
        if out_of_range:
            st.warning(f"{len(out_of_range)} value(s) are outside the training range, so the prediction "
                       "may be unreliable: " + ", ".join(out_of_range[:5]) + ("…" if len(out_of_range) > 5 else ""))
else:
    st.info("Adjust the values (or load an example in the sidebar) and press **Predict**.")
