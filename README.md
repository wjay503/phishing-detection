# Explainable Phishing Detection Prototype

This Streamlit application implements the revised project:

**AI-Based Phishing Website Detection and Explainable Risk Classification System**

## Main technologies
- XGBoost — phishing classifier
- SHAP TreeExplainer — local prediction explanations
- Streamlit — interactive user interface

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally opens the application at:

```text
http://localhost:8501
```

## Prototype workflow

Website Features
→ Data Validation / Input Controls
→ XGBoost Classifier
→ Phishing Probability
→ SHAP Explanation
→ Risk Classification
→ Human Review

## Outputs

For every submitted example the app displays:

1. Prediction: Legitimate or Phishing
2. Phishing probability
3. Risk level
4. Recommendation
5. SHAP waterfall explanation
6. Top 10 local feature contributions

## Risk bands

- Low Risk: probability < 40%
- Suspicious / Review Required: 40% to < 75%
- High Risk: >= 75%

The underlying machine-learning model remains a binary classifier. The three risk
bands are interface-level decision-support categories.

## Important limitation

The system uses the 30 already-engineered characteristics in the UCI Phishing
Websites dataset. It does not automatically extract these values from a raw URL.
A production system would require a separate feature-extraction component.

## Responsible AI note

SHAP explains which feature values influenced the XGBoost prediction. SHAP
contributions should not be interpreted as causal evidence. Final decisions remain
with the user or security analyst.


## Streamlit caching fix

The SHAP explainer cache function uses `_model` rather than `model` as its
parameter name. Streamlit interprets leading-underscore parameters as values
that should not be hashed. This avoids `UnhashableParamError` for the
`XGBClassifier` instance while still caching the SHAP TreeExplainer.
