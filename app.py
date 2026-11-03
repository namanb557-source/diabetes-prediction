import os
import urllib.request
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

app = Flask(__name__)

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
MODEL = None
MODEL_METRICS = {}

def load_dataset():
    # Download a publicly available, commonly used Pima Indians diabetes teaching dataset.
    df = pd.read_csv(DATA_URL)
    required = FEATURES + ["Outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Dataset columns missing: " + ", ".join(missing))
    df = df[required].copy()
    # In this dataset, zero is used for some measurements where a value is missing.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df.loc[df[col] == 0, col] = np.nan
    df = df.dropna(subset=["Outcome"])
    return df

def train_model():
    global MODEL, MODEL_METRICS
    df = load_dataset()
    X = df[FEATURES]
    y = df["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    MODEL = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    MODEL.fit(X_train, y_train)
    predictions = MODEL.predict(X_test)
    probabilities = MODEL.predict_proba(X_test)[:, 1]
    MODEL_METRICS = {
        "accuracy": round(accuracy_score(y_test, predictions) * 100, 1),
        "auc": round(roc_auc_score(y_test, probabilities), 3),
        "rows": len(df)
    }

try:
    train_model()
except Exception as exc:
    # App still starts and displays a useful message if dataset download is temporarily unavailable.
    print("Model training failed:", exc)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}
    if request.method == "POST":
        try:
            if MODEL is None:
                raise RuntimeError("The ML model is not ready. Please refresh in a moment.")
            labels = {
                "Pregnancies": "Number of pregnancies",
                "Glucose": "Glucose (mg/dL)",
                "BloodPressure": "Diastolic blood pressure (mmHg)",
                "SkinThickness": "Skin thickness (mm)",
                "Insulin": "Insulin (mu U/mL)",
                "BMI": "BMI",
                "DiabetesPedigreeFunction": "Diabetes pedigree function",
                "Age": "Age (years)"
            }
            for key, label in labels.items():
                raw = request.form.get(key, "").strip()
                if raw == "":
                    raise ValueError(f"Please enter {label.lower()}.")
                values[key] = float(raw)
                if not np.isfinite(values[key]):
                    raise ValueError(f"Please enter a valid value for {label.lower()}.")
            if not 0 <= values["Pregnancies"] <= 20: raise ValueError("Pregnancies must be between 0 and 20.")
            if not 20 <= values["Glucose"] <= 350: raise ValueError("Glucose must be between 20 and 350.")
            if not 20 <= values["BloodPressure"] <= 200: raise ValueError("Blood pressure must be between 20 and 200.")
            if not 0 <= values["SkinThickness"] <= 100: raise ValueError("Skin thickness must be between 0 and 100.")
            if not 0 <= values["Insulin"] <= 1000: raise ValueError("Insulin must be between 0 and 1000.")
            if not 10 <= values["BMI"] <= 80: raise ValueError("BMI must be between 10 and 80.")
            if not 0 <= values["DiabetesPedigreeFunction"] <= 3: raise ValueError("Diabetes pedigree function must be between 0 and 3.")
            if not 1 <= values["Age"] <= 120: raise ValueError("Age must be between 1 and 120.")

            row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
            probability = float(MODEL.predict_proba(row)[0, 1])
            prediction = int(MODEL.predict(row)[0])
            result = {
                "prediction": prediction,
                "probability": round(probability * 100, 1),
                "label": "Model predicts positive" if prediction == 1 else "Model predicts negative",
                "class_name": "positive" if prediction == 1 else "negative"
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Something went wrong while processing the inputs. Please check the values and try again."
    return render_template("index.html", result=result, error=error, values=values, metrics=MODEL_METRICS)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "metrics": MODEL_METRICS}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
