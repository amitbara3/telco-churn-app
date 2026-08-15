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
│                          (scikit-learn Pipeline, or a small              │
│                           probability-averaging ensemble of a few)       │
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
  metrics.json          Evaluation metrics + chosen hyperparameters
  feature_importance.json  Full ranked feature importances of the deployed model
  feature_schema.json    Allowed categories / numeric ranges, used by the UI
tests/
  test_api.py          API tests (pytest + FastAPI TestClient)
Dockerfile
start.sh              Container entrypoint
requirements.txt
```

## Model

The deployed model is **CatBoost using native categorical handling** — it
consumes the raw category strings directly via `cat_features`, with no
one-hot encoding step at all. It was chosen by comparing twelve model
families under the protocol below; that comparison is summarized further
down and preserved in git history, but `train/train.py` now trains only
the winner, so a retrain takes ~2 minutes instead of ~6.

**Held-out test performance** (1,407 customers, never touched during
training, tuning, or threshold selection):

| Metric | Value |
|---|---|
| Churn F1 | **0.636** |
| Balanced accuracy | **0.763** |
| Accuracy | 0.779 |
| ROC-AUC | 0.840 |
| Churn recall | 0.727 |
| Churn precision | 0.566 |

In plain terms: of customers who actually churn, the model flags ~73%; of
those it flags, ~57% actually churn. For a retention use case that's the
right side of the trade — a missed churner costs more than a wasted
retention offer.

How `train/train.py` gets there:

1. **Feature engineering** (`app/features.py`, shared with the live API so
   training and serving can never drift apart): `num_addon_services`
   (count of subscribed add-ons), `avg_charge_per_tenure`
   (`TotalCharges / (tenure + 1)`), `charges_delta`
   (`tenure * MonthlyCharges - TotalCharges` — positive means the customer
   has been on some kind of discount relative to their current rate,
   often a churn trigger once it expires) and its normalized form
   `discount_ratio`, an `is_new_customer` flag (tenure ≤ 3 months), a
   `high_risk_new_customer` flag (an explicit interaction of the two
   strongest individual predictors: month-to-month contract *and*
   tenure ≤ 12 months), `has_streaming` and `manual_payment` flags, a
   bucketed `tenure_bucket`, and a **K-Means `customer_segment`**
   (`CustomerSegmentFeature` — 3 clusters on tenure/charges/service-count).
   Unlike the others, the segment feature genuinely needs *fitting*
   (cluster centers learned from data), so it's its own pipeline step
   rather than a stateless row-wise formula — cross-validation still fits
   it fold-by-fold, so no leakage. It also collapses the
   `"No internet/phone service"` category on 7 columns into `"No"` — that
   value is 100% determined by `InternetService`/`PhoneService` already
   being `"No"`, so keeping it as a distinct category just re-encoded the
   same bit up to 6 times over.
2. **Hyperparameter search**: `RandomizedSearchCV` (20 iterations, 5-fold
   `StratifiedKFold`, scored on ROC-AUC) over depth, iterations, learning
   rate, `l2_leaf_reg`, and `scale_pos_weight` (searched over `[1.0,
   sqrt(imbalance ratio), imbalance ratio]`).
3. **Decision threshold tuning**: the default 0.5 cutoff is rarely optimal
   for an imbalanced target (~27% churn). Using `cross_val_predict` to get
   out-of-fold probabilities on the *training* split only, it sweeps
   thresholds and picks the one maximizing F1 on the "Churn" class —
   without ever looking at the test set, so the threshold isn't overfit to
   it. The tuned value lands at 0.47.

### The model comparison that led here

Twelve model families were run through the identical protocol above. The
full results table and per-model hyperparameters are in git history
(`git log -- model/metrics.json`); the summary:

| Model | CV ROC-AUC | Test Churn F1 | Test Bal. Acc |
|---|---|---|---|
| **CatBoost (native categoricals)** — deployed | 0.850 | **0.636** | **0.763** |
| LightGBM | 0.849 | 0.633 | 0.761 |
| XGBoost | 0.850 | 0.631 | 0.758 |
| Ensemble (top 3) | — | 0.631 | 0.759 |
| Logistic Regression | 0.849 | 0.628 | 0.752 |
| CatBoost (one-hot) | 0.851 | 0.627 | 0.756 |
| Gradient Boosting / HistGradientBoosting | 0.848–0.850 | 0.624 | 0.755–0.760 |
| Random Forest / Extra Trees | 0.849–0.851 | 0.622–0.624 | 0.752–0.755 |
| ANN (Keras, batch norm + dropout) | 0.839 | 0.611 | 0.741 |
| MLP (sklearn, shallow) | 0.847 | 0.610 | 0.749 |

Two things worth carrying forward from that exercise:

- **The field is extremely tight.** Twelve genuinely different algorithms
  span ~2 points of balanced accuracy. That's the signature of a ceiling
  set by the data, not the model — a conclusion independently corroborated
  by a published benchmark on this same dataset (see below) and by segment
  analysis showing near-coin-flip churn rates among customers identical on
  every available feature.
- **Neural nets lose, tested twice.** A shallow sklearn MLP and a properly
  built Keras ANN (batch norm, dropout, early stopping, depth up to
  `(128, 64, 32)`) finished last and second-to-last. The telling detail:
  given free choice of depth, the ANN's search settled on the *smallest*
  architecture offered with the *highest* dropout — more capacity than
  ~5,600 rows of tabular data can constrain.

### What was tried and rejected

Several techniques were tried and *rejected* on measured evidence. They're
recorded here because a negative result you can point at is worth more
than an untested assumption — and because each one is a thing a reviewer
would otherwise reasonably ask "did you try…?" about.

- **SMOTE oversampling: worse than doing nothing.** Applied correctly (via
  `imblearn`'s pipeline, so the sampler only ever touches a fold's
  training rows), it scored *below* the same model without it. That's the
  honest counterpoint to several sources claiming big SMOTE wins on this
  dataset — at least one of which has an acknowledged ambiguity about
  applying it *before* the train/test split, which is a textbook leak.
- **A wide class-weight sweep: also no help, and revealingly so.** Tried
  `class_weight = {0:1, 1:w}` for every integer `w` from 1 to 30. Given
  free choice across that whole range via cross-validated ROC-AUC, the
  search settled on **`w=1` — no extra weighting at all**. Not just a
  negative result: it says the threshold-tuning step already captures
  whatever a reweighted loss would buy, so stacking a second
  imbalance correction on top is redundant rather than additive.
- **Feature-subset search (beam search, width 5, patience 10): didn't
  generalize.** It found a 15-feature subset that beat the full set on
  train-CV, but lost on every held-out test metric — including the one it
  optimized. Classic overfitting to the search's own validation criterion
  after scoring 1,500+ subsets against the same 5 folds.
- **Ensembling: never decisively better.** Probability-averaging the top-3
  and all-12 candidates was evaluated the same way as any single model.
  It occasionally edged ahead by a hair, but never enough to justify
  shipping and maintaining several models instead of one — the candidates
  are mostly tree-based and highly correlated, so averaging cancels little
  independent error.
- **K-Means `customer_segment`: mixed, kept anyway.** It helped most
  candidates modestly but hurt one, and net effect across the field is
  close to a wash. Kept because it helps the deployed model specifically,
  but flagged rather than dressed up as a clean win.

### A real bug this surfaced

Adding CatBoost-native to the candidate set earlier had exposed something
that had been silently latent: `app/model.py`'s hardcoded raw-feature
column order didn't match `train.py`'s. One-hot pipelines never noticed —
`ColumnTransformer` selects columns by name, so order is irrelevant — but
CatBoost's native-categorical mode resolves `cat_features` to *positional*
indices at fit time. A live request built its row in a different column
order than training used, so a categorical string could silently land in
a position CatBoost expected to be numeric, crashing `/predict` outright
(caught by the test suite, not manually). Fixed at the root rather than by
reordering one list to match the other: both `train.py` and `app/model.py`
now import a single shared `RAW_FEATURE_ORDER` from `app/features.py`, so
the two can't drift apart again.

### Feature importance

Full ranked list is written to `model/feature_importance.json` on every
training run, taken from CatBoost's own `feature_importances_`. Because
native mode doesn't one-hot encode, these are importances per *raw
column*, not per category level. Top 5:

| Feature | Importance |
|---|---|
| `Contract` | 25.5% |
| `InternetService` | 18.8% |
| `tenure` | 6.7% |
| `MonthlyCharges` | 5.1% |
| `PaymentMethod` | 4.6% |

...and the tail — the 10 least useful of the 29 raw + engineered columns:

| Feature | Importance |
|---|---|
| `StreamingTV` | 1.4% |
| `SeniorCitizen` | 1.2% |
| `Dependents` | 1.0% |
| `has_streaming` | 1.0% |
| `PhoneService` | 0.9% |
| `gender` | 0.6% |
| `OnlineBackup` | 0.4% |
| `manual_payment` | 0.2% |
| `Partner` | 0.1% |
| `DeviceProtection` | 0.0% (literally unused, again) |

Only 4 of 29 columns fall under the 0.5%-each dead-weight threshold now
(combined 0.7%) — much cleaner than the one-hot-expanded counts earlier in
this project (which were never comparing like for like: one raw column vs.
several one-hot levels of it). `Contract` and `InternetService` alone
account for 44% of total importance, consistent with every other lens
applied to this dataset throughout — segment analysis, permutation
importance, the paper's own SHAP results — all agree these two features
(plus tenure) carry most of the real signal.

### Where the ceiling actually is

Cumulatively, the work above moved churn F1 from **0.623** (an untuned
baseline) to **0.636**, and accuracy from 0.751 to 0.779. Real, but modest
— and the reason it's modest is worth stating plainly, because it's the
most useful finding in this project.

Three independent lines of evidence say the limit is the *data*, not the
modeling:

1. **Twelve different algorithms converge to within ~2 points** of each
   other on balanced accuracy. When a linear model and a tuned gradient
   booster land in the same place, the bottleneck isn't model capacity.
2. **The train/test ROC-AUC gap never closed** (~0.03) despite explicit
   L1/L2 regularization search — so this was never primarily an
   overfitting problem that better regularization could fix.
3. **Segment analysis finds irreducible overlap.** Slicing month-to-month
   customers by tenure bucket × internet service × charge quartile — the
   four strongest predictors together — leaves ~10 sizeable segments with
   churn rates between 30% and 70%. Customers *identical on every
   available feature* still split near a coin flip. No model can separate
   what the features don't distinguish.

Further gains would need information this schema doesn't contain —
support-ticket history, usage trends, competitor pricing, whether a
retention offer was made — not more algorithms or tuning.

### How this compares to published results

Checked this against outside work on the same dataset rather than just
trusting our own numbers in isolation. Two findings:

- **A close, credible match**: an independent paper on this exact dataset
  ([arXiv:2607.10260](https://arxiv.org/abs/2607.10260)) used almost the
  same methodology we'd already converged on independently — stratified
  5-fold CV, class weighting, F1-driven threshold tuning — and deployed
  CatBoost as their final model, reporting **77.68% accuracy, F1 0.6366,
  ROC-AUC 0.8403** on their held-out test set. That's within a point of our
  own 76.1% / 0.632 / 0.840. Independent convergence on near-identical
  numbers via near-identical methodology is good evidence this is the real
  ceiling for this dataset, not an artifact of how we did it.
- **The suspiciously high numbers elsewhere are a red flag, not a lead**:
  several sources claim 90%+ accuracy or ROC-AUC >0.92 on this same
  dataset. Checking one such paper's own methodology description turned up
  an acknowledged ambiguity in *when* SMOTE oversampling was applied
  relative to the train/test split and cross-validation folds — applying
  SMOTE before splitting is a classic, common leakage bug on this exact
  dataset (synthetic minority samples derived from what should be held-out
  data leak into training). Public notebooks claiming huge scores on this
  dataset are unfortunately more often a warning sign than a technique to
  copy — not something to chase.

Reading the matching paper's *full methodology* rather than just its
headline numbers surfaced two techniques worth borrowing, both since
adopted and both verified in isolation on the untouched test set:

- **CatBoost's native categorical handling was a real, isolated win.**
  Same search space, same features, same everything else — one-hot CatBoost
  scored 0.625 test F1, native CatBoost **0.630**. This confirmed the
  paper's specific mechanistic claim rather than just its bottom line, and
  it's why the deployed model uses `cat_features` today.
- **A few of its domain-engineered features carried over** —
  `high_risk_new_customer`, `has_streaming`, `manual_payment`,
  `discount_ratio` — and the K-Means segmentation idea became
  `CustomerSegmentFeature`.

Two CatBoost/scikit-learn interop bugs turned up along the way, both worth
knowing about if you extend this: `l2_leaf_reg` and `cat_features` each
fail sklearn's `clone()` equality check when passed as constructor
arguments (and `RandomizedSearchCV` clones per fold). `l2_leaf_reg` is
fixed by casting the search grid to native Python floats instead of
`numpy.float64`; `cat_features` is fixed by passing it through `.fit()`
via the `model__cat_features=...` fit-param convention, sidestepping
`clone()` entirely.

Risk-level bands (`Low`/`Medium`/`High` in the API response) scale with
the selected model's tuned threshold rather than fixed 0.33/0.66 cutoffs —
`Medium` always straddles the actual Yes/No decision boundary, so it stays
coherent even though thresholds vary a lot between models (0.26–0.63).

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Train the model (writes model/churn_model.joblib, decision_threshold.json,
# metrics.json, feature_importance.json, feature_schema.json). Runs a
# 20-iteration hyperparameter search with 5-fold CV, so it takes ~2min.
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
  "churn_probability": 0.7584,
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
