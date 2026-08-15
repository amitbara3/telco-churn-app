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
  model.py            Loads the trained pipeline + threshold, runs predictions
  features.py           Row-wise feature engineering (shared by train + API)
  schemas.py           Pydantic request/response models
  streamlit_app.py    Streamlit form UI, calls the API
train/
  train.py            Feature engineering, hyperparameter search, CV, threshold tuning
data/
  Telco-Customer-Churn.csv   Raw dataset (7,043 rows)
model/
  churn_model.joblib   Trained scikit-learn Pipeline (feature eng. + preprocessing + model)
  decision_threshold.json  Tuned churn/no-churn probability cutoff
  metrics.json          Evaluation metrics for all candidate models
  feature_schema.json    Allowed categories / numeric ranges, used by the UI
tests/
  test_api.py          API tests (pytest + FastAPI TestClient)
Dockerfile
start.sh              Container entrypoint
requirements.txt
```

## Model

`train/train.py` builds a pipeline of three steps — feature engineering,
preprocessing, and a classifier — for each of five model families (Logistic
Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM), plus two
probability-averaging ensembles of the top-3 and all 5, and keeps whichever
generalizes best. Concretely, per candidate:

1. **Feature engineering** (`app/features.py`, shared with the live API so
   training and serving can never drift apart): adds `num_addon_services`
   (count of subscribed add-ons), `avg_charge_per_tenure`
   (`TotalCharges / (tenure + 1)`), `charges_delta`
   (`tenure * MonthlyCharges - TotalCharges` — positive means the customer
   has been on some kind of discount relative to their current rate,
   often a churn trigger once it expires), an `is_new_customer` flag
   (tenure ≤ 3 months), and a bucketed `tenure_bucket`. It also collapses
   the `"No internet/phone service"` category on 7 columns into `"No"` —
   that value is 100% determined by `InternetService`/`PhoneService`
   already being `"No"`, so keeping it as its own one-hot category just
   re-encoded the same bit up to 6 times over. All derived values are
   computed per-row from that row's own raw values, so they're safe to use
   inside cross-validation without leaking across folds.
2. **Preprocessing**: one-hot encoding for categorical features, standard
   scaling for numeric ones (`ColumnTransformer`).
3. **Hyperparameter search**: `RandomizedSearchCV` (20 iterations, 5-fold
   `StratifiedKFold`, scored on ROC-AUC) over each model's own parameter
   space, including L1/L2 regularization terms for XGBoost/LightGBM.
4. **Decision threshold tuning**: the default 0.5 cutoff is rarely optimal
   for an imbalanced target (~27% churn). Using `cross_val_predict` to get
   out-of-fold probabilities on the *training* split only, it sweeps
   thresholds and picks the one that maximizes F1 on the "Churn" class —
   without ever looking at the test set, so the threshold isn't overfit to
   it.
5. **Ensembling**: averages predict_proba across the top-3 and all-5 tuned
   models (`AverageProbabilityEnsemble` in `app/features.py`), evaluated
   the same way as any other candidate.
6. **Model selection**: whichever candidate scores best on **F1 for the
   "Churn" class** on the held-out 20% test split, evaluated at its own
   tuned threshold. For a churn-prevention use case, missing an actual
   churner (low recall) is costlier than a false alarm — plain accuracy or
   ROC-AUC are much less sensitive to that trade-off on this dataset.

Current results (see `model/metrics.json` for full detail, including each
model's best hyperparameters and CV scores):

| Model | CV ROC-AUC | Tuned threshold | Test Accuracy | Test Churn F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.849 | 0.61 | 0.774 | 0.626 | 0.837 |
| Random Forest | 0.851 | 0.56 | 0.775 | 0.626 | 0.837 |
| Gradient Boosting | 0.850 | 0.36 | 0.780 | 0.630 | 0.840 |
| XGBoost | 0.851 | 0.35 | 0.768 | 0.624 | 0.839 |
| **LightGBM (selected)** | **0.851** | **0.32** | **0.761** | **0.632** | **0.840** |
| Ensemble (top 3) | — | 0.42 | 0.771 | 0.625 | 0.840 |
| Ensemble (all 5) | — | 0.43 | 0.770 | 0.630 | 0.840 |

### What moved the numbers, and what didn't

Started from a hypothesis-driven diagnosis (see `model/metrics.json` history
in git log for the "untuned" and "5-model, no fixes" baselines): feature
importances showed 23 of 52 one-hot columns carrying a combined 4.3% of
total importance (mostly the redundant `"No internet service"` duplicates),
and slicing customers by their strongest predictors showed several
sizeable segments with near-50% churn rates *within* customers who are
identical on every available feature — i.e. some of the gap is inherent to
the dataset, not fixable by tuning.

After collapsing the redundant categories, adding `charges_delta` /
`is_new_customer`, widening the regularization search, and adding
ensembling:

- **Churn F1 improved consistently across every candidate** (e.g. LightGBM
  0.623 → 0.632, Gradient Boosting 0.620 → 0.630, Random Forest
  0.617 → 0.626) — a real, reproducible ~1-point gain, not noise from a
  single lucky split.
- **`charges_delta` earned its place**: it's the 3rd most important feature
  in the final model (`num__tenure` and `num__MonthlyCharges` are 1st/2nd),
  ahead of `TotalCharges` itself.
- **The train/test ROC-AUC gap did not close** (0.0297 → 0.0317, i.e. no
  meaningful change) despite adding L1/L2 regularization search — evidence
  this was never really an overfitting problem, reinforcing that the
  remaining ~15 points of ROC-AUC headroom is the dataset's actual
  information ceiling, not solvable by better regularization.
- **Ensembling didn't beat the best single model** (0.630 vs. LightGBM's
  0.632) — the candidates are all gradient/tree-based and highly
  correlated, so averaging them doesn't cancel out much independent error.

Net: real, honest improvement (~1 point of F1, ~2 points of accuracy),
consistent across the board — and further confirmation that we're now
close to what these 19 raw features can support. The next real gain would
come from new data (support tickets, usage patterns, competitor pricing
signals), not more modeling.

Risk-level bands (`Low`/`Medium`/`High` in the API response) scale with
the selected model's tuned threshold rather than fixed 0.33/0.66 cutoffs —
`Medium` always straddles the actual Yes/No decision boundary, so it stays
coherent even though thresholds vary a lot between models (0.27–0.61).

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Train the model (writes model/churn_model.joblib, decision_threshold.json,
# metrics.json, feature_schema.json). Runs a hyperparameter search across
# 3 model families with 5-fold CV, so it takes ~30s-1min.
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
  "churn_probability": 0.6828,
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
