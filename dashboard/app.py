import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc
)

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# =====================
# TITLE
# =====================
st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption("Advanced ML Monitoring System")

# =====================
# LOAD MODEL
# =====================
model_path = os.path.join("models", "model.pkl")

if not os.path.exists(model_path):
    st.error("Model not found. Train model first.")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
default_threshold = bundle["threshold"]

# =====================
# SIDEBAR
# =====================
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload creditcard.csv")

threshold = st.sidebar.slider(
    "Decision Threshold",
    0.1, 0.9, float(default_threshold)
)

show_fraud_only = st.sidebar.checkbox("Show Only Fraud")

# =====================
# DATA LOADING
# =====================
if uploaded:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
    else:
        df = pd.read_csv("data/sample_creditcard.csv")

# =====================
# FEATURES
# =====================
X = df.drop("Class", axis=1)
y = df["Class"]

probs = model.predict_proba(X)[:, 1]
preds = (probs > threshold).astype(int)

df["Fraud Probability"] = probs
df["Prediction"] = preds

# =====================
# RISK SEGMENT
# =====================
def risk(p):
    if p > 0.8:
        return "High"
    elif p > 0.4:
        return "Medium"
    else:
        return "Low"

df["Risk"] = df["Fraud Probability"].apply(risk)

# =====================
# KPI METRICS
# =====================
total = len(df)
frauds = int(preds.sum())
fraud_rate = (frauds / total) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", total)
col2.metric("Fraud Alerts", frauds)
col3.metric("Fraud %", f"{fraud_rate:.2f}")

# =====================
# MODEL PERFORMANCE
# =====================
precision = precision_score(y, preds)
recall = recall_score(y, preds)
f1 = f1_score(y, preds)

c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{precision:.2f}")
c2.metric("Recall", f"{recall:.2f}")
c3.metric("F1 Score", f"{f1:.2f}")

# =====================
# TABS
# =====================
tab1, tab2, tab3 = st.tabs(["📊 Monitoring", "📈 Analytics", "🧠 Insights"])

# =====================
# TAB 1 - DATA
# =====================
with tab1:
    if show_fraud_only:
        st.dataframe(df[df["Prediction"] == 1])
    else:
        st.dataframe(df)

# =====================
# TAB 2 - ANALYTICS
# =====================
with tab2:
    colA, colB = st.columns(2)

    # Distribution
    with colA:
        st.subheader("Fraud Probability Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(probs, bins=50, ax=ax1)
        st.pyplot(fig1)

    # Confusion Matrix
    with colB:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax3.legend()
    st.pyplot(fig3)

    # PR Curve
    st.subheader("Precision-Recall Curve")
    precision_c, recall_c, _ = precision_recall_curve(y, probs)

    fig4, ax4 = plt.subplots()
    ax4.plot(recall_c, precision_c)
    st.pyplot(fig4)

# =====================
# TAB 3 - INSIGHTS
# =====================
with tab3:
    st.subheader("Risk Breakdown")
    risk_counts = df["Risk"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("Top Risk Transactions")
    st.dataframe(df.sort_values(by="Fraud Probability", ascending=False).head(20))

    st.subheader("Single Transaction Test")

    if st.button("Test Random Transaction"):
        sample = X.sample(1)
        prob = model.predict_proba(sample)[0][1]
        result = "FRAUD" if prob > threshold else "SAFE"

        st.write({
            "fraud_probability": float(prob),
            "decision": result
        })