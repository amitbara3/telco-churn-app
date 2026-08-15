---
title: Telco Customer Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Telco Customer Churn Predictor

An end-to-end deployed ML app: a scikit-learn model trained on the Kaggle
["Telco Customer Churn"](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset, served through a FastAPI backend, with a Streamlit UI on top —
all in a single Docker container.

**Live demo:** _add your Hugging Face Space / Render URL here after deploying_

## Architecture

```
┌─────────────────────────── Docker container ───────────────────────────┐
│                                                                          │
│   Streamlit UI (port 7860)  ── HTTP ──►  FastAPI service (port 8000)    │
│   app/streamlit_app.py                   app/main.py                   │
│                                              │                          │
│                                              ▼                          │
│                                   model/churn_model.joblib              │
│                                   (scikit-learn Pipeline)                │
└──────────────────────────────────────────────────────────────────────┘
```

`start.sh` launches the FastAPI server in the background and the Streamlit
app in the foreground. Streamlit calls the API over `localhost` — it's not
just calling the model directly, so the API is independently usable
(e.g. via `curl`, `/docs`, or another client) even though the two share a
container for free, one-service hosting on Spaces.

## Project structure

```
app/
  main.py            FastAPI app (/health, /predict)
  model.py            Loads the trained pipeline, runs predictions
  schemas.py           Pydantic request/response models
  streamlit_app.py    Streamlit form UI, calls the API
train/
  train.py            Loads data, trains + evaluates 3 models, saves the best
data/
  Telco-Customer-Churn.csv   Raw dataset (7,043 rows)
model/
  churn_model.joblib   Trained scikit-learn Pipeline (preprocessing + model)
  metrics.json          Evaluation metrics for all candidate models
  feature_schema.json    Allowed categories / numeric ranges, used by the UI
tests/
  test_api.py          API tests (pytest + FastAPI TestClient)
Dockerfile
start.sh              Container entrypoint
requirements.txt
```

## Model

`train/train.py` builds a `ColumnTransformer` (one-hot encoding for
categorical features, standard scaling for numeric features) feeding into
one of three classifiers, and keeps whichever scores best on **F1 for the
"Churn" class** on a held-out 20% test split. Churn is imbalanced (~27% of
customers), and for a churn-prevention use case, missing an actual churner
(low recall) is costlier than a false alarm — so selection uses F1 on the
positive class rather than plain accuracy or ROC-AUC, which are far less
sensitive to that trade-off on this dataset.

Current results (see `model/metrics.json` for full detail):

| Model | Accuracy | Churn F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.726 | 0.607 | 0.835 |
| **Random Forest (selected)** | **0.751** | **0.623** | **0.834** |
| Gradient Boosting | 0.795 | 0.579 | 0.838 |

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Train the model (writes model/churn_model.joblib, metrics.json, feature_schema.json)
python train/train.py

# Run the tests
python -m pytest tests/ -q

# Run the API
uvicorn app.main:app --reload

# In another terminal, run the UI
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

API docs (Swagger UI) are then at http://localhost:8000/docs.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

```json
{
  "churn_probability": 0.71,
  "churn_prediction": "Yes",
  "risk_level": "High"
}
```

## Docker

```bash
docker build -t telco-churn-app .
docker run -p 7860:7860 -p 8000:8000 telco-churn-app
```

Then open http://localhost:7860 for the UI, or http://localhost:8000/docs
for the API.

## Deploying to Hugging Face Spaces

This repo is ready to push straight to a Space — the README frontmatter
above (`sdk: docker`, `app_port: 7860`) is exactly what Spaces reads to
configure the build.

1. Create a new Space at https://huggingface.co/new-space with **Docker**
   as the SDK.
2. Add it as a git remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push hf main
   ```
3. The Space will build the Dockerfile and come up at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

## Deploying to Render instead

Render can build the same `Dockerfile` directly (New → Web Service → point
at this repo → Environment: Docker). Render expects the service to listen
on `$PORT`; either set the Streamlit/uvicorn ports from that env var, or
simplest for an API-only deploy, change the Dockerfile's `CMD` to just:

```dockerfile
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(also mirrored by IBM as sample data for the same dataset) — 7,043 customers,
19 features (demographics, account info, services subscribed), binary churn
label.
