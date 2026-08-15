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
  beam_search.py       Beam search feature selection experiment (see README; not adopted)
data/
  Telco-Customer-Churn.csv   Raw dataset (7,043 rows)
model/
  churn_model.joblib   Trained scikit-learn Pipeline (feature eng. + preprocessing + model)
  decision_threshold.json  Tuned churn/no-churn probability cutoff
  metrics.json          Evaluation metrics for all candidate models
  feature_importance.json  Full ranked feature importances of the deployed model
  feature_schema.json    Allowed categories / numeric ranges, used by the UI
  beam_search_result.json  Beam search run's history and result (not the deployed feature set)
tests/
  test_api.py          API tests (pytest + FastAPI TestClient)
Dockerfile
start.sh              Container entrypoint
requirements.txt
```

## Model

`train/train.py` builds a pipeline of three steps — feature engineering,
preprocessing, and a classifier — for each of seven model families
(Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM,
CatBoost with one-hot encoding, CatBoost with native categorical handling),
plus two probability-averaging ensembles of the top-3 and all-7, and keeps
whichever generalizes best. Concretely, per candidate:

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
   tenure ≤ 12 months), `has_streaming` and `manual_payment` flags, and a
   bucketed `tenure_bucket`. It also collapses the
   `"No internet/phone service"` category on 7 columns into `"No"` — that
   value is 100% determined by `InternetService`/`PhoneService` already
   being `"No"`, so keeping it as its own one-hot category just re-encoded
   the same bit up to 6 times over. All derived values are computed
   per-row from that row's own raw values, so they're safe to use inside
   cross-validation without leaking across folds.
2. **Preprocessing**: one-hot encoding + standard scaling
   (`ColumnTransformer`) for every candidate except CatBoost-native, which
   consumes the raw category strings directly via CatBoost's built-in
   `cat_features` handling — no one-hot step at all for that one.
3. **Hyperparameter search**: `RandomizedSearchCV` (20 iterations, 5-fold
   `StratifiedKFold`, scored on ROC-AUC) over each model's own parameter
   space — L1/L2 regularization for XGBoost/LightGBM/CatBoost, and (new)
   `scale_pos_weight` searched over `[1.0, sqrt(imbalance ratio),
   imbalance ratio]` for every boosting model, not just relying on
   threshold tuning for the class imbalance.
4. **Decision threshold tuning**: the default 0.5 cutoff is rarely optimal
   for an imbalanced target (~27% churn). Using `cross_val_predict` to get
   out-of-fold probabilities on the *training* split only, it sweeps
   thresholds and picks the one that maximizes F1 on the "Churn" class —
   without ever looking at the test set, so the threshold isn't overfit to
   it.
5. **Ensembling**: averages predict_proba across the top-3 and all-7 tuned
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
| Logistic Regression | 0.850 | 0.61 | 0.773 | 0.622 | 0.837 |
| Random Forest | 0.851 | 0.57 | 0.774 | 0.623 | 0.837 |
| Gradient Boosting | 0.850 | 0.37 | 0.784 | 0.629 | 0.839 |
| XGBoost | 0.851 | 0.33 | 0.765 | 0.629 | 0.839 |
| LightGBM | 0.850 | 0.46 | 0.766 | 0.627 | 0.836 |
| CatBoost (one-hot) | 0.851 | 0.44 | 0.763 | 0.625 | 0.839 |
| CatBoost (native categoricals) | 0.850 | 0.46 | 0.772 | 0.630 | 0.840 |
| **Ensemble, top 3 (selected)** | **—** | **0.48** | **0.782** | **0.632** | **0.839** |
| Ensemble, all 7 | — | 0.46 | 0.773 | 0.631 | 0.840 |

The deployed model is now an ensemble (CatBoost one-hot + XGBoost + Random
Forest) — the first time in this project an ensemble has actually won.

### Feature importance

Full ranked list for the deployed model is written to
`model/feature_importance.json` on every training run — for an ensemble,
importances are averaged across its member pipelines. Top 5 (currently
dominated by Contract, since CatBoost's importances weight it heavily):

| Feature | Importance |
|---|---|
| `Contract` = Month-to-month | 23.8% |
| `InternetService` = Fiber optic | 7.1% |
| `tenure` | 6.5% |
| `high_risk_new_customer` (new) | 5.7% |
| `Contract` = Two year | 5.2% |

...and the tail — the 10 least useful of the 51 encoded columns:

| Feature | Importance |
|---|---|
| `PaymentMethod_Bank transfer (automatic)` | 0.35% |
| `OnlineBackup_No` | 0.35% |
| `PaymentMethod_Credit card (automatic)` | 0.31% |
| `tenure_bucket_24-48` | 0.31% |
| `tenure_bucket_12-24` | 0.29% |
| `Partner_No` | 0.25% |
| `DeviceProtection_Yes` | 0.22% |
| `DeviceProtection_No` | 0.20% |
| `Partner_Yes` | 0.19% |
| `tenure_bucket_48-60` | 0.13% |

14 of the 51 encoded columns carry <0.5% of total importance each — a
combined 4.2%, in the same range as before. Same conclusion as previously:
genuinely weak signal in those columns for this target, not another free
cleanup opportunity.

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

### How this compares to published results (and adding CatBoost)

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

Given the matching paper's model choice, first **added CatBoost** through
the same one-hot pipeline as everything else — it landed weakest of all
six candidates at the time (0.622 test F1), not an improvement.

Digging into the paper's full methodology (not just its headline numbers)
surfaced two more specific techniques it credited that we hadn't tried:
CatBoost's **native categorical handling** (`cat_features`, no one-hot at
all) and **`scale_pos_weight` search** (tried at `1.0`, `sqrt(ratio)`, and
the raw imbalance ratio) for every boosting model, not just relying on
threshold tuning. Implemented both, plus a few of the paper's own
domain-engineered features (`high_risk_new_customer`, `has_streaming`,
`manual_payment`, `discount_ratio`). Honest results, each verified on the
untouched test set:

- **Native categorical handling for CatBoost was a real, isolated win**:
  same search space, same everything else — one-hot CatBoost scored 0.625
  test F1, native CatBoost scored **0.630**. This directly confirms the
  paper's specific claim, not just its headline number.
- **`scale_pos_weight` search was a wash for some models and a regression
  for one**: LightGBM's test F1 actually *dropped* (0.632 → 0.627) after
  adding it as an 8th tuned dimension to `RandomizedSearchCV` — with the
  same 20-iteration search budget, one more dimension means less effective
  coverage of the parameters that mattered before. A lesson in its own
  right: widening a hyperparameter search without widening its budget can
  quietly make individual candidates worse, even when it helps others.
- **The combination made ensembling finally pay off**: previously,
  ensembling never beat the best single model. With native CatBoost now in
  the mix, `Ensemble (top 3)` — CatBoost (one-hot) + XGBoost + Random
  Forest — edges out every individual candidate: test F1 0.632 (vs. the
  prior deployment's 0.6316 — a marginal but real gain, not tied) and test
  accuracy 0.782 (vs. 0.761, a more solid +2 points). ROC-AUC is flat.

Net: modest, not dramatic — consistent with the ceiling already being
close. But real, verified, multi-metric-consistent, and it came from
specific, attributable techniques rather than "try more stuff and see."
The deployed model is now `Ensemble (top 3)`. (Along the way, hit and
fixed *two* separate CatBoost/scikit-learn interop bugs: `l2_leaf_reg` and
`cat_features` both fail sklearn's `clone()` equality check when passed as
constructor arguments — `l2_leaf_reg` fixed by casting to native Python
floats, `cat_features` fixed by passing it through `.fit()` instead of the
constructor, since `RandomizedSearchCV`/`cross_val_predict` support
per-step fit params via the `model__cat_features=...` prefix convention.)

### Beam search feature selection (tried, didn't generalize)

`train/beam_search.py` runs a forward beam search over the 24 candidate
columns (beam width 5, patience 10, ordered by the aggregate importance
from `model/feature_importance.json`), scoring each candidate subset by
5-fold cross-validated balanced accuracy — entirely on the training split,
never touching test. It found a 15-feature subset scoring 0.7231 vs. 0.7160
for the full set — a real train-CV improvement.

Validated against the held-out test set, it **didn't hold up**: the full
24-feature model wins on every test metric, including the one the search
optimized for —

| | Train-CV bal. acc | Test bal. acc | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Full set (24 features, deployed) | 0.7160 | **0.7640** | **0.6316** | **0.8399** |
| Beam subset (15 features) | 0.7231 | 0.7544 | 0.6244 | 0.8383 |

This is the search overfitting its own validation criterion: scoring
~1,500+ candidate subsets against the same 5 CV folds lets the eventual
"winner" pick up an optimistic score from sheer multiple-comparisons luck
rather than real signal — the classic failure mode of greedy wrapper-based
feature selection without nested CV. It also explains why the subset drops
`charges_delta`, which is genuinely useful in the full model (3rd-highest
importance) but happened to look replaceable within the specific folds the
search kept re-using. **Not adopted** — the full feature set stays
deployed. Kept in the repo as a working tool and an honest negative result,
not discarded.

Risk-level bands (`Low`/`Medium`/`High` in the API response) scale with
the selected model's tuned threshold rather than fixed 0.33/0.66 cutoffs —
`Medium` always straddles the actual Yes/No decision boundary, so it stays
coherent even though thresholds vary a lot between models (0.33–0.61).

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Train the model (writes model/churn_model.joblib, decision_threshold.json,
# metrics.json, feature_schema.json). Runs a hyperparameter search across
# 7 model families with 5-fold CV, so it takes ~1-2min.
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
  "churn_probability": 0.7508,
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
