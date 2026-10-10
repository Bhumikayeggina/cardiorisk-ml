"""
CardioRisk - interactive demonstration of the final model.

Run locally (from the repo root, venv active):
    streamlit run app/streamlit_app.py

Academic demonstration only: NOT a medical device and NOT a diagnosis.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # so `import src...` works everywhere

from src.explain import explain                    # noqa: E402

st.set_page_config(page_title="CardioRisk", page_icon="❤️", layout="centered")

PLACEHOLDER = "Select..."
INCOME_LABELS = ["Less than $10,000", "$10,000 to less than $15,000",
                 "$15,000 to less than $20,000", "$20,000 to less than $25,000",
                 "$25,000 to less than $35,000", "$35,000 to less than $50,000",
                 "$50,000 to less than $75,000", "$75,000 to less than $100,000",
                 "$100,000 to less than $150,000", "$150,000 to less than $200,000",
                 "$200,000 or more"]
YES_NO = [("No", 0), ("Yes", 1)]

# column -> (short name, question text, options [(label, model code)] or None for the slider)
QUESTIONS = {
    "_AGE_G":   ("Age group", "What is your age group?",
                 [("18 to 24", 1), ("25 to 34", 2), ("35 to 44", 3), ("45 to 54", 4),
                  ("55 to 64", 5), ("65 or older", 6)]),
    "SEXVAR":   ("Sex", "What is your sex?", [("Male", 1), ("Female", 2)]),
    "GENHLTH":  ("General health", "Would you say your general health is...",
                 [("Excellent", 1), ("Very good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)]),
    "DIABETE4": ("Diabetes", "Have you ever been told you have diabetes?",
                 [("Yes", 1), ("Yes, but only during pregnancy", 2), ("No", 3),
                  ("No, pre-diabetes or borderline", 4)]),
    "HAVARTH4": ("Arthritis", "Told you have arthritis, gout, lupus or fibromyalgia?", YES_NO),
    "_SMOKER3": ("Smoking", "Which best describes your smoking?",
                 [("Current smoker, every day", 1), ("Current smoker, some days", 2),
                  ("Former smoker", 3), ("Never smoked", 4)]),
    "CHCCOPD3": ("COPD", "Told you have COPD, emphysema or chronic bronchitis?", YES_NO),
    "DIFFWALK": ("Difficulty walking", "Serious difficulty walking or climbing stairs?", YES_NO),
    "CHECKUP1": ("Last checkup", "How long since your last routine checkup?",
                 [("Within the past year", 1), ("Within the past 2 years", 2),
                  ("Within the past 5 years", 3), ("5 or more years ago", 4), ("Never", 8)]),
    "INCOME3":  ("Income", "What is your annual household income?",
                 [(lab, i + 1) for i, lab in enumerate(INCOME_LABELS)]),
    "PHYSHLTH": ("Poor physical health days", "Days in the past 30 your physical health was not good",
                 None),
    "CHCKDNY2": ("Kidney disease", "Told you have kidney disease (not kidney stones)?", YES_NO),
    "MARITAL":  ("Marital status", "What is your marital status?",
                 [("Married", 1), ("Divorced", 2), ("Widowed", 3), ("Separated", 4),
                  ("Never married", 5), ("Member of an unmarried couple", 6)]),
    "CHCOCNC1": ("Other cancer", "Told you had any cancer other than skin cancer?", YES_NO),
}

LAYOUT = [("About you", ["_AGE_G", "SEXVAR", "MARITAL", "INCOME3"]),
          ("Your health", ["GENHLTH", "PHYSHLTH", "DIFFWALK", "CHECKUP1", "_SMOKER3"]),
          ("Conditions you have been told you have",
           ["DIABETE4", "HAVARTH4", "CHCCOPD3", "CHCKDNY2", "CHCOCNC1"])]


@st.cache_resource
def load_assets():
    pipe = joblib.load(ROOT / "models" / "final_xgb_pipeline.joblib")
    features = json.loads((ROOT / "reports" / "final_features.json").read_text())["final"]
    decision = json.loads((ROOT / "reports" / "final_decision.json").read_text())
    bands = json.loads((ROOT / "reports" / "risk_bands.json").read_text())
    return pipe, features, decision, bands


def test_metric(name):
    try:
        tbl = pd.read_csv(ROOT / "reports" / "final_test_results.csv", index_col=0)
        return float(tbl.loc[name, "estimate"])
    except Exception:
        return None


pipe, FEATURES, decision, bands = load_assets()
T_HIGH, T_MOD = decision["flag_threshold"], decision["moderate_threshold"]
PREV = bands["prevalence"]

st.title("❤️ CardioRisk")
st.caption("Estimates how likely someone with these answers is to report coronary heart disease "
           "or a heart attack, using a model trained on the 2024 CDC BRFSS survey.")
st.warning("Academic demonstration only. This is **not** a medical device or a diagnosis, and it "
           "cannot replace advice from a health professional. Nothing you enter is stored.")

answers, labels = {}, {}
with st.form("risk_form"):
    for section, cols in LAYOUT:
        st.subheader(section)
        left, right = st.columns(2)
        for i, col in enumerate(cols):
            short, text, options = QUESTIONS[col]
            box = left if i % 2 == 0 else right
            with box:
                if options is None:
                    value = st.slider(text, 0, 30, 0, key=col)
                    answers[col], labels[col] = float(value), f"{value} days"
                else:
                    choice = st.selectbox(text, [PLACEHOLDER] + [o[0] for o in options], key=col)
                    if choice != PLACEHOLDER:
                        answers[col] = float(dict(options)[choice])
                        labels[col] = choice
    submitted = st.form_submit_button("Estimate risk", type="primary")

if submitted:
    missing = [QUESTIONS[c][0] for c in FEATURES if c not in answers]
    if missing:
        st.error("Please answer every question. Still missing: " + ", ".join(missing) + ".")
    else:
        row = pd.DataFrame([{c: answers[c] for c in FEATURES}], columns=FEATURES)
        p = float(pipe.predict_proba(row)[:, 1][0])
        band = "Low" if p < T_MOD else ("Moderate" if p < T_HIGH else "High")
        info = bands["bands"][band]

        st.header("Result")
        c1, c2 = st.columns(2)
        c1.metric("Estimated risk", f"{p:.1%}", f"{p / PREV:.1f}x the survey average ({PREV:.1%})",
                  delta_color="off")
        c2.metric("Risk band", band)
        st.write(f"**{band} band.** In cross-validated training data, {info['observed_rate']:.1%} of "
                 f"people in this band reported heart disease; the band contains "
                 f"{info['share_of_people']:.0%} of people and {info['share_of_cases']:.0%} of all cases.")
        if band == "High":
            st.info("This score is above the model's screening threshold. In testing, a cutoff like "
                    "this caught about 80% of reported cases while flagging about a third of people, "
                    "so most flagged people do not have the condition.")

        agg, base, _ = explain(pipe, row, FEATURES)
        contrib = agg.iloc[0].sort_values()
        names = [f"{QUESTIONS[c][0]}: {labels[c]}" for c in contrib.index]

        st.subheader("What drove this estimate")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(names, contrib.values, color=["#d62728" if v > 0 else "#1f77b4" for v in contrib.values])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("effect on risk (log-odds; right = higher risk, left = lower)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        up = contrib[contrib > 0.05].sort_values(ascending=False).head(3)
        down = contrib[contrib < -0.05].head(3)
        if len(up):
            st.write("**Raising the estimate:** " + "; ".join(
                f"{QUESTIONS[c][0]} ({labels[c]}, x{np.exp(v):.2f} odds)" for c, v in up.items()))
        if len(down):
            st.write("**Lowering the estimate:** " + "; ".join(
                f"{QUESTIONS[c][0]} ({labels[c]}, x{np.exp(v):.2f} odds)" for c, v in down.items()))
        st.caption("Explanations show what the model learned from survey data; they describe "
                   "associations, not causes.")

with st.expander("About the model and its limits"):
    auc, rec = test_metric("roc_auc"), test_metric("recall")
    if auc is not None and rec is not None:
        st.write(f"On a sealed test set the model reached ROC-AUC {auc:.2f}; the screening "
                 f"threshold finds about {rec:.0%} of reported cases.")
    st.markdown(
        "- Trained on the 2024 CDC BRFSS survey (self-reported, \"ever told\" heart disease status); "
        "it estimates the likelihood of that status, not a clinical diagnosis.\n"
        "- It cannot see blood pressure or cholesterol (not in the 2024 file) or any clinical measurement.\n"
        "- Risk estimates are well calibrated overall and within sex, age and income groups, but "
        "slightly over-estimate risk for some racial/ethnic groups, and at one cutoff people under 45 "
        "are rarely flagged because heart disease is rare at those ages.\n"
        "- Please answer every question; a blank answer is not treated as neutral by the model.")
