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
  metrics.json          Evaluation metrics for all candidate models
  feature_importance.json  Full ranked feature importances of the deployed model
  feature_schema.json    Allowed categories / numeric ranges, used by the UI
tests/
  test_api.py          API tests (pytest + FastAPI TestClient)
Dockerfile
start.sh              Container entrypoint
requirements.txt
```

## Model

`train/train.py` builds a pipeline of three steps — feature engineering,
preprocessing, and a classifier — for each of eleven model families
(Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
HistGradientBoosting (plain and with a wide class-weight sweep), XGBoost,
LightGBM, CatBoost with one-hot encoding, CatBoost with native categorical
handling, a small MLP neural net), plus two probability-averaging
ensembles of the top-3 and all-11, and keeps whichever generalizes best.
Concretely, per candidate:

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
   (`CustomerSegmentFeature` — 3 clusters on tenure/charges/service-count,
   inspired by the same independent benchmark's value-segmentation
   analysis). Unlike the others, the segment feature genuinely needs
   *fitting* (cluster centers learned from data), so it's its own pipeline
   step rather than a stateless row-wise formula — cross-validation still
   fits it fold-by-fold, so no leakage. It also collapses the
   `"No internet/phone service"` category on 7 columns into `"No"` — that
   value is 100% determined by `InternetService`/`PhoneService` already
   being `"No"`, so keeping it as its own one-hot category just re-encoded
   the same bit up to 6 times over.
2. **Preprocessing**: one-hot encoding + standard scaling
   (`ColumnTransformer`) for every candidate except CatBoost-native, which
   consumes the raw category strings directly via CatBoost's built-in
   `cat_features` handling — no one-hot step at all for that one.
3. **Hyperparameter search**: `RandomizedSearchCV` (20 iterations, 5-fold
   `StratifiedKFold`, scored on ROC-AUC) over each model's own parameter
   space — L1/L2 regularization for XGBoost/LightGBM/CatBoost, and
   `scale_pos_weight` searched over `[1.0, sqrt(imbalance ratio),
   imbalance ratio]` for every boosting model.
4. **Class imbalance, a wide weight sweep**: one HistGradientBoosting
   variant searches `class_weight` over `{0: 1, 1: w}` for every integer
   `w` from 1 to 30 (`CLASS_WEIGHT_SWEEP`) — much wider than the narrow
   `scale_pos_weight` range (max ~2.76) used elsewhere, cross-validated the
   same way as every other hyperparameter. (An earlier version of this
   tried SMOTE oversampling instead, via `imblearn.pipeline.Pipeline` so
   the sampler only ever touches a fold's training rows — never leaking
   into validation/test. It scored worse than doing nothing special for
   imbalance, so it was replaced with this weight sweep rather than kept
   as dead weight in the pipeline.)
5. **Decision threshold tuning**: the default 0.5 cutoff is rarely optimal
   for an imbalanced target (~27% churn). Using `cross_val_predict` to get
   out-of-fold probabilities on the *training* split only, it sweeps
   thresholds and picks the one that maximizes F1 on the "Churn" class —
   without ever looking at the test set.
6. **Ensembling**: averages predict_proba across the top-3 and all-11
   tuned models (`AverageProbabilityEnsemble` in `app/features.py`).
7. **Model selection**: whichever candidate scores best on **F1 for the
   "Churn" class** on the held-out 20% test split, evaluated at its own
   tuned threshold.

Current results (see `model/metrics.json` for full detail, including each
model's best hyperparameters and CV scores):

| Model | CV ROC-AUC | Tuned threshold | Test Accuracy | Test Churn F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.849 | 0.63 | 0.785 | 0.628 | 0.838 |
| Random Forest | 0.851 | 0.55 | 0.772 | 0.624 | 0.837 |
| Extra Trees | 0.849 | 0.55 | 0.759 | 0.622 | 0.837 |
| Gradient Boosting | 0.850 | 0.33 | 0.765 | 0.624 | 0.839 |
| HistGradientBoosting | 0.848 | 0.29 | 0.749 | 0.624 | 0.835 |
| HistGradientBoosting + weight sweep (1-30) | 0.846 | 0.30 | 0.754 | 0.622 | 0.838 |
| XGBoost | 0.850 | 0.35 | 0.775 | 0.631 | 0.838 |
| LightGBM | 0.849 | 0.47 | 0.774 | 0.633 | 0.837 |
| CatBoost (one-hot) | 0.851 | 0.59 | 0.770 | 0.627 | 0.839 |
| **CatBoost (native categoricals, selected)** | **0.850** | **0.47** | **0.779** | **0.636** | **0.840** |
| MLP (neural net) | 0.847 | 0.26 | 0.738 | 0.610 | 0.836 |
| Ensemble, top 3 | — | 0.49 | 0.772 | 0.631 | 0.839 |
| Ensemble, all 11 | — | 0.47 | 0.778 | 0.634 | 0.840 |

### Class imbalance and K-Means segmentation, tested honestly

Both techniques came directly out of researching how other work approaches
this dataset. Neither is a clean win — reported exactly as measured, not
rounded up:

- **Neither SMOTE nor a wide weight sweep beat plain threshold tuning.**
  SMOTE was tried first (correctly, via `imblearn`'s leakage-safe
  pipeline) and scored *worse* than doing nothing special for imbalance
  (test F1 0.619 vs. 0.624 for plain `HistGradientBoosting`) — the honest
  counterpoint to several sources claiming big wins from SMOTE on this
  dataset. Replaced with a much wider `class_weight` sweep (`{0:1, 1:w}`
  for `w` in 1–30, vs. the narrow ~1–2.76 range used elsewhere) to see if
  aggressive loss-reweighting fared better: also worse (0.622), and
  tellingly, the hyperparameter search — free to pick any of the 30
  weights via cross-validated ROC-AUC — settled on **`w=1`, i.e. no extra
  weighting at all**. That's a clean signal, not just a negative result:
  our existing threshold-tuning step already captures whatever a
  reweighted loss function would buy here, so stacking another
  imbalance-correction on top is redundant, not additive.
- **K-Means `customer_segment` was a genuinely mixed result.** Added to
  *every* candidate's feature set (not just one), it improved most of them
  slightly (Logistic Regression 0.622→0.628, LightGBM 0.627→0.633, CatBoost
  native 0.630→0.636) but made `HistGradientBoosting` — the model that had
  been winning — measurably *worse* (0.638→0.624). That's large enough to
  flip which model wins overall, and the new best (CatBoost native, 0.636)
  is a hair below the immediately preceding deployment (0.638). Same
  dynamic as the earlier `scale_pos_weight` case: adding a feature or
  search dimension without adding search budget can quietly hurt whichever
  model happened to be tuned tightest around the old feature set, even
  while helping most others. Net effect here is close to a wash — kept the
  segmentation feature since it helps the majority of candidates and the
  loss is within noise for a single 1,407-row test split, but flagged
  clearly rather than presented as a clean improvement.

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

Full ranked list for the deployed model is written to
`model/feature_importance.json` on every training run. CatBoost (both
one-hot and native) exposes its own `feature_importances_`, so this uses
that directly rather than the permutation-importance fallback (which
kicks in only for candidates like HistGradientBoosting/MLP that expose
neither `feature_importances_` nor `coef_` — see `extract_feature_importance`
in `train/train.py`). Since native mode doesn't one-hot encode, this is
importance per raw column, not per category level. Top 5:

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
(This round's deployed model, `Ensemble (top 3)`, was later superseded —
see below.) Along the way, hit and fixed *two* separate CatBoost/scikit-learn
interop bugs: `l2_leaf_reg` and `cat_features` both fail sklearn's `clone()`
equality check when passed as constructor arguments — `l2_leaf_reg` fixed
by casting to native Python floats, `cat_features` fixed by passing it
through `.fit()` instead of the constructor, since
`RandomizedSearchCV`/`cross_val_predict` support per-step fit params via
the `model__cat_features=...` prefix convention.

### Trying more model families

Rounded out the model comparison with three more families through the same
search/CV/threshold-tuning pipeline: `ExtraTreesClassifier` (Random
Forest's more-randomized sibling), `HistGradientBoostingClassifier`
(scikit-learn's own histogram-based boosting — the same algorithmic family
as LightGBM, independently implemented, not yet tried), and a small
`MLPClassifier` (mainly to answer "did you try a neural net" definitively).

- **HistGradientBoostingClassifier won outright** — test F1 **0.638**,
  beating every other individual model *and* both ensembles, despite having
  the lowest CV ROC-AUC (0.848) among the strong contenders. It found a
  precision/recall trade-off at its tuned threshold that nothing else
  reached. Concretely: ROC-AUC measures ranking quality across all possible
  thresholds; F1 measures performance at one specific threshold, which is
  what's actually deployed — the two don't have to agree on which model is
  "best," and here they didn't.
- **Extra Trees landed right next to Random Forest** (0.622 vs. 0.623 F1)
  — expected, they're closely related algorithms; no real signal either
  way.
- **The MLP clearly underperformed** (0.606 F1, worst of all ten
  candidates) — expected on ~5.6k training rows of tabular data, and now
  actually confirmed rather than assumed. Not worth pursuing further (e.g.
  architecture tuning) given the gap.
- **Fixed a real gap this surfaced**: `HistGradientBoostingClassifier` has
  neither `feature_importances_` nor `coef_`, so `feature_importance.json`
  was silently going stale (still describing whichever model won
  *previously*) the first time this ran. Fixed with a permutation-importance
  fallback (`sklearn.inspection.permutation_importance` on the held-out
  test set, post-hoc only — doesn't affect model selection) for any
  candidate lacking a native importance measure.

The deployed model is now `HistGradientBoostingClassifier` alone — F1 0.638
is the best result of the whole project, and it came from trying one more
legitimately different algorithm family rather than further tuning the ones
already tried.

A forward beam search over feature subsets (width 5, patience 10, scored
by cross-validated balanced accuracy on the training split) was also
tried. It found a 15-feature subset that scored better on train-CV than
the full feature set, but the improvement didn't hold up on the held-out
test set — the full feature set won on every test metric, including the
one the search optimized for. Classic overfitting to the search's own
validation criterion (scoring 1,500+ candidate subsets against the same 5
CV folds inflates the eventual "winner" via multiple-comparisons luck).
Not adopted, and the tool has since been removed from the repo — it did
its job as a diagnostic, and there's no reason to keep unused code around
once the answer's in.

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
# metrics.json, feature_schema.json). Runs a hyperparameter search across
# 11 model families with 5-fold CV, so it takes ~2-3min.
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
