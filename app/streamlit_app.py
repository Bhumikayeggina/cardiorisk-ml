"""
Phase 13: Interactive CardioRisk demo.
Run locally with:  streamlit run app/streamlit_app.py
Deploy at: https://share.streamlit.io (connect this GitHub repo)
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="CardioRisk", page_icon="❤️", layout="centered")

st.title("❤️ CardioRisk — Cardiovascular Risk Screening")
st.caption(
    "Academic demonstration only — trained on self-reported population survey data "
    "(CDC BRFSS 2024). This is **not** a medical diagnostic tool."
)

# --- Load trained pipeline ---
# TODO: point this at your actual saved pipeline from Phase 7/9/11
# pipeline = joblib.load("models/final_pipeline.pkl")
# explainer = joblib.load("models/shap_explainer.pkl")

st.subheader("Enter your information")

col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("Age group", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    general_health = st.selectbox("General health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
with col2:
    smoker = st.selectbox("Smoking status", ["Never", "Former", "Current"])
    physically_active = st.selectbox("Physically active (past 30 days)", ["Yes", "No"])
    sleep_hours = st.slider("Average sleep hours", 3, 12, 7)
    diabetes = st.selectbox("Told you have diabetes", ["Yes", "No"])

if st.button("Estimate risk", type="primary"):
    # TODO: assemble the input row in the exact column order/format the
    # pipeline's ColumnTransformer expects, then:
    #
    # input_df = pd.DataFrame([{ ... }])
    # proba = pipeline.predict_proba(input_df)[0, 1]
    # st.metric("Estimated CVD risk", f"{proba:.1%}")
    #
    # shap_values = explainer(input_df)
    # st.subheader("Why this estimate")
    # shap.plots.waterfall(shap_values[0])  # via streamlit-shap's st_shap()

    st.info("Wire this up once your trained pipeline is saved to models/ (see PROJECT_GUIDE.md Phase 13).")

st.divider()
st.caption("Built with Scikit-learn, XGBoost, and SHAP · [GitHub repo link here]")
