from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

app = Flask(__name__)
app.secret_key = os.environ.get("RETAINIQ_SECRET_KEY", "change-this-local-development-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}

MODEL = None
METRICS = {}

def load_resources():
    global MODEL, METRICS
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("churn_model.pkl is missing from the project folder.")
        MODEL = joblib.load(MODEL_PATH)
    if not METRICS and METRICS_PATH.exists():
        METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return MODEL

def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"

def get_probability(frame: pd.DataFrame) -> np.ndarray:
    model = load_resources()
    # The saved sklearn Pipeline includes the same preprocessing used during training.
    probabilities = model.predict_proba(frame[FEATURES])[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0, 1)

def make_recommendations(customer: dict, risk: str) -> list[str]:
    tips = []
    if customer["Contract"] == "Month-to-month":
        tips.append("Offer a relevant annual-plan comparison or loyalty benefit.")
    if float(customer["MonthlyCharges"]) >= 80:
        tips.append("Review plan value and explain lower-cost options that preserve needed services.")
    if int(customer["tenure"]) <= 12:
        tips.append("Prioritize onboarding support and a proactive satisfaction check-in.")
    if customer["PaymentMethod"] == "Electronic check":
        tips.append("Offer an easy switch to automatic payment if the customer prefers it.")
    if not tips:
        tips.append("Maintain regular engagement and monitor service experience.")
    if risk == "High":
        tips.insert(0, "Prioritize a personal retention conversation; understand the customer's reason before offering incentives.")
    elif risk == "Medium":
        tips.insert(0, "Schedule a timely check-in and monitor engagement signals.")
    return tips

def clean_customer(form) -> dict:
    customer = {}
    for field in FEATURES:
        raw = form.get(field, "").strip()
        if raw == "":
            raise ValueError(f"Please complete the {field} field.")
        if field in ("SeniorCitizen", "tenure"):
            customer[field] = int(raw)
        elif field in ("MonthlyCharges", "TotalCharges"):
            customer[field] = float(raw)
        else:
            customer[field] = raw
    if customer["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1.")
    if customer["tenure"] < 0 or customer["tenure"] > 100:
        raise ValueError("Tenure must be between 0 and 100 months.")
    if customer["MonthlyCharges"] < 0 or customer["TotalCharges"] < 0:
        raise ValueError("Charges cannot be negative.")
    return customer

@app.context_processor
def inject_globals():
    return {"current_year": 2026}

@app.route("/")
def dashboard():
    try:
        load_resources()
        status = "Online"
        error = None
    except Exception as exc:
        status = "Unavailable"
        error = str(exc)
    return render_template(
        "index.html",
        page="dashboard",
        metrics=METRICS,
        model_status=status,
        model_error=error,
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_values = {}
    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            customer = clean_customer(request.form)
            probability = float(get_probability(pd.DataFrame([customer]))[0])
            risk = risk_level(probability)
            result = {
                "probability": probability,
                "percent": round(probability * 100, 1),
                "risk": risk,
                "recommendations": make_recommendations(customer, risk),
                "customer": customer,
            }
            session["last_prediction"] = {
                "probability": probability,
                "risk": risk,
                "customer": customer,
            }
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("predict.html", page="predict", options=OPTIONS,
                           features=FEATURES, values=form_values, result=result)

@app.route("/batch", methods=["GET", "POST"])
def batch():
    summary = None
    table_html = None
    if request.method == "POST":
        upload = request.files.get("file")
        if not upload or not upload.filename:
            flash("Choose a CSV file to upload.", "error")
        else:
            try:
                df = pd.read_csv(upload)
                missing = [c for c in FEATURES if c not in df.columns]
                if missing:
                    raise ValueError("CSV is missing required columns: " + ", ".join(missing))
                probabilities = get_probability(df[FEATURES].copy())
                output = df.copy()
                output["Churn Probability (%)"] = (probabilities * 100).round(1)
                output["Risk Level"] = [risk_level(float(p)) for p in probabilities]
                output["Suggested Action"] = [
                    "Priority retention outreach" if r == "High"
                    else "Proactive check-in" if r == "Medium"
                    else "Continue standard engagement"
                    for r in output["Risk Level"]
                ]
                counts = output["Risk Level"].value_counts().to_dict()
                summary = {
                    "total": len(output),
                    "high": int(counts.get("High", 0)),
                    "medium": int(counts.get("Medium", 0)),
                    "low": int(counts.get("Low", 0)),
                }
                session["batch_csv"] = output.to_csv(index=False)
                table_html = output.head(100).to_html(
                    classes="data-table", index=False, border=0,
                    escape=True
                )
            except Exception as exc:
                flash(f"Could not process this file: {exc}", "error")
    return render_template("batch.html", page="batch", summary=summary, table_html=table_html)

@app.route("/download-batch")
def download_batch():
    content = session.get("batch_csv")
    if not content:
        flash("Run a batch prediction before downloading a report.", "error")
        return redirect(url_for("batch"))
    return send_file(
        io.BytesIO(content.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="RetainIQ_customer_risk_report.csv",
    )

@app.route("/model")
def model():
    load_resources()
    return render_template("model.html", page="model", metrics=METRICS)

@app.route("/sample-csv")
def sample_csv():
    rows = [{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.9, "TotalCharges": 1078.8
    }]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FEATURES)
    writer.writeheader()
    writer.writerows(rows)
    return send_file(io.BytesIO(buffer.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name="retainiq_batch_template.csv")

@app.route("/about")
def about():
    return render_template("about.html", page="about")

@app.errorhandler(413)
def too_large(_error):
    flash("The uploaded file is too large. Maximum size is 8 MB.", "error")
    return redirect(url_for("batch"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
