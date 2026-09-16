
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "phishing_xgboost_shap_model.joblib"
SCHEMA_PATH = APP_DIR / "feature_schema.json"
EXAMPLES_PATH = APP_DIR / "example_inputs.json"

st.set_page_config(
    page_title="Explainable Phishing Risk Classifier",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_explainer(_model):
    # Leading underscore tells Streamlit not to hash this argument.
    # XGBClassifier objects are not hashable by Streamlit's cache system.
    return shap.TreeExplainer(_model)

bundle = load_model_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]
feature_schema = load_json(SCHEMA_PATH)
examples = load_json(EXAMPLES_PATH)
explainer = load_explainer(model)

def initialise_feature_state():
    for feature in feature_names:
        if feature not in st.session_state:
            st.session_state[feature] = feature_schema[feature][0]

def load_selected_example():
    selected = st.session_state.get("example_selector", "Manual input")
    if selected == "Manual input":
        return
    values = examples[selected]
    for feature in feature_names:
        st.session_state[feature] = values[feature]

def reset_inputs():
    for feature in feature_names:
        st.session_state[feature] = feature_schema[feature][0]
    st.session_state["example_selector"] = "Manual input"

initialise_feature_state()

st.title("🛡️ AI-Based Phishing Website Detection")
st.subheader("Explainable Risk Classification with XGBoost + SHAP")

st.info(
    "This academic prototype uses 30 already-extracted website characteristics. "
    "It does not inspect a raw live URL directly. Predictions are decision support, "
    "not a guarantee that a website is safe or malicious."
)

with st.sidebar:
    st.header("System information")
    st.write("**Model:** XGBoost")
    st.write("**Explainability:** SHAP TreeExplainer")
    st.write("**Positive class:** Phishing")
    st.write("**Input features:** 30")
    st.write("**Human review:** Required")

    st.divider()
    st.subheader("Load a test example")
    st.selectbox(
        "Example",
        ["Manual input"] + list(examples.keys()),
        key="example_selector",
        on_change=load_selected_example
    )
    st.button("Reset inputs", on_click=reset_inputs)

st.subheader("1. Website characteristics")
st.caption(
    "Select the encoded values used by the UCI Phishing Websites dataset."
)

cols = st.columns(3)
for idx, feature in enumerate(feature_names):
    options = feature_schema[feature]
    if st.session_state.get(feature) not in options:
        st.session_state[feature] = options[0]
    with cols[idx % 3]:
        st.selectbox(
            feature.replace("_", " ").title(),
            options=options,
            key=feature
        )

st.divider()

if st.button("Analyse website", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[st.session_state[f] for f in feature_names]],
        columns=feature_names
    )

    prediction = int(model.predict(input_df)[0])
    phishing_probability = float(model.predict_proba(input_df)[0, 1])

    if phishing_probability >= 0.75:
        risk_level = "High Risk"
        recommendation = (
            "Treat this website as suspicious. Do not enter credentials or sensitive "
            "information until it has been independently verified."
        )
    elif phishing_probability >= 0.40:
        risk_level = "Suspicious / Review Required"
        recommendation = (
            "The model is uncertain. Perform additional verification before trusting "
            "or interacting with the website."
        )
    else:
        risk_level = "Low Risk"
        recommendation = (
            "The model considers this website more likely to be legitimate, but the "
            "result does not guarantee that the website is safe."
        )

    label = "Phishing" if prediction == 1 else "Legitimate"

    st.subheader("2. Prediction and risk assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", label)
    c2.metric("Phishing probability", f"{phishing_probability * 100:.2f}%")
    c3.metric("Risk level", risk_level)

    st.progress(float(np.clip(phishing_probability, 0, 1)))

    if risk_level == "High Risk":
        st.error(recommendation)
    elif risk_level == "Suspicious / Review Required":
        st.warning(recommendation)
    else:
        st.success(recommendation)

    st.subheader("3. SHAP explanation")
    st.caption(
        "SHAP explains how each feature influenced this prediction. "
        "Positive contributions push the model toward Phishing; negative contributions "
        "push it toward Legitimate."
    )

    shap_values = explainer(input_df)
    explanation = shap_values[0]

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf(), clear_figure=True)

    values = explanation.values
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": input_df.iloc[0].values,
        "SHAP Contribution": values
    })
    contrib_df["Absolute Contribution"] = contrib_df["SHAP Contribution"].abs()
    contrib_df = contrib_df.sort_values("Absolute Contribution", ascending=False)

    st.subheader("4. Top feature contributions")
    st.dataframe(
        contrib_df.head(10)[
            ["Feature", "Feature Value", "SHAP Contribution"]
        ],
        use_container_width=True
    )

    st.caption(
        "Important: SHAP values explain the model's behaviour. They do not prove "
        "that a feature caused the website to be phishing or legitimate."
    )

    with st.expander("View all submitted feature values"):
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True
        )

st.divider()
st.caption(
    "Academic prototype — AI-Based Phishing Website Detection and Explainable Risk "
    "Classification System. XGBoost + SHAP, with human-in-the-loop decision support."
)
