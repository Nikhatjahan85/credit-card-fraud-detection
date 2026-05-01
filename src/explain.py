import shap
import joblib
import pandas as pd

bundle = joblib.load("models/model.pkl")
model = bundle["model"]

def explain_sample(sample):
    df = pd.DataFrame([sample])

    explainer = shap.Explainer(model.named_steps["model"])
    shap_values = explainer(df)

    return shap_values