import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import load_data, split_xy, basic_cleaning
from pipeline import get_pipeline

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Put creditcard.csv in data folder.")

df = load_data(DATA_PATH)
df = basic_cleaning(df)

X, y = split_xy(df)

# =========================
# HANDLE IMBALANCE
# =========================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# =========================
# PIPELINE
# =========================
preprocessor = get_pipeline(X)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric="logloss"
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# =========================
# TRAIN
# =========================
pipe.fit(X_train, y_train)

# =========================
# FIND BEST THRESHOLD
# =========================
probs = pipe.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.4f}")

# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump({
    "model": pipe,
    "threshold": float(best_threshold)
}, MODEL_PATH)

print("Model saved successfully!")