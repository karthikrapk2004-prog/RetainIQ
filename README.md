# RetainIQ — Customer Churn Prediction

A professional Flask web application built around the supplied scikit-learn customer churn pipeline.

## Included features
- Overview dashboard with supplied model metrics
- Single-customer churn probability assessment
- Low / Medium / High risk bands (same thresholds as the original app: 40% and 70%)
- General retention action prompts
- CSV batch scoring and downloadable risk report
- Model comparison and confusion-matrix view
- Sample CSV template download
- Responsive desktop and mobile interface

## Project structure
```text
RetainIQ_Professional/
├── app.py
├── churn_model.pkl
├── model_metrics.json
├── requirements.txt
├── run_windows.bat
├── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── batch.html
│   ├── model.html
│   └── about.html
└── static/
    ├── style.css
    └── app.js
```

## Run on Windows (PowerShell)

1. Extract this ZIP to a folder.
2. Open that folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the website:

```powershell
python app.py
```

7. Open `http://127.0.0.1:5000`.

You can also double-click `run_windows.bat` after completing dependency installation.

## Model and data notes
- The original `churn_model.pkl` is copied unchanged.
- The saved pipeline includes preprocessing and a Logistic Regression classifier.
- The application sends the same 19 input feature columns used by the original training script to the saved pipeline.
- `model_metrics.json` is the supplied metrics file. Reported values are displayed as provided; the website does not retrain the model.
- The original Telco churn CSV was not included in the uploaded files. Therefore this version does not fabricate customer totals, churn rates, or a customer-level historical dashboard. Use Batch Analysis to upload a CSV with the 19 required model input columns.
- The batch template provides a valid example row. Replace it with permitted customer data before using batch prediction.
- Risk bands are operational labels, not calibrated probability guarantees. Review thresholds and validate on current data before production use.

## Required CSV columns for batch scoring
`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

An optional `customerID` column is retained in the exported output but is not used by the model.

## Security and deployment
This is a local demonstration app. Before public deployment, configure a strong `RETAINIQ_SECRET_KEY`, add authentication/authorization, HTTPS, secure file handling, audit logging, privacy controls, and operational monitoring. Do not upload personal customer information unless you have permission and an approved environment.
