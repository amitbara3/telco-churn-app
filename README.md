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
  main.py            FastAPI app (/health, /predict, /model-info)
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
  test_api.py          HTTP-layer tests (pytest + FastAPI TestClient)
  test_pipeline.py     Model/pipeline tests, incl. train-vs-serve feature parity
  test_streamlit_ui.py Headless UI tests (Streamlit AppTest) of the submit path
.github/workflows/
  ci.yml               Tests, a real training run, and a Docker build + container smoke test
Dockerfile
start.sh              Container entrypoint
requirements.txt      Direct dependencies (what you edit)
requirements.lock     All 68 packages pinned + hashed, linux/py3.11 — what Docker installs
```

## What CI actually verifies

Three independent jobs, because each covers a failure the others miss:

| Job | Catches |
|---|---|
| **Tests** (46) | API contract, train/serve feature parity, calibration not silently dropped, the Streamlit submit path |
| **Training** | A broken `train/train.py`. The test suite exercises the saved *artifact*, so training code could break and stay green until someone next retrained. Runs for real at a reduced search budget. |
| **Docker** | That the deployment target builds, boots, and answers on `/health`, `/predict` and the UI — plus reports image size |

## Operating it

- **`GET /health`** — liveness. `start.sh` and the Docker `HEALTHCHECK`
  both probe this, and the CI container test waits on it.
- **`GET /model-info`** — what a running instance is actually serving:
  model name, calibration method, decision threshold, and its held-out
  metrics. Lets you confirm a deployment picked up the model you think it
  did without shelling into the container.
- **`POST /predict`** — one JSON line per prediction is written to stdout:

  ```json
  {"event": "prediction", "churn_probability": 0.6804, "churn_prediction": "Yes",
   "risk_level": "High", "tenure_bucket": "0-12", "latency_ms": 39.21}
  ```

  Enough to watch the output distribution and latency for drift, which is
  what silently degrades a churn model once the calibration stops matching
  reality. Deliberately **no customer attributes** — the request body is
  personal data, and `tenure` is coarsened to a band rather than logged
  raw. A test asserts the payload contains exactly these keys, so this
  can't quietly grow to include PII.

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
| Balanced accuracy | **0.764** |
| Accuracy | 0.776 |
| ROC-AUC | 0.846 |
| Churn recall | 0.738 |
| Churn precision | 0.559 |
| Brier score | 0.136 |
| Expected calibration error | 0.020 |

In plain terms: of customers who actually churn, the model flags ~74%; of
those it flags, ~56% actually churn. For a retention use case that's the
right side of the trade — a missed churner costs more than a wasted
retention offer.

The last two rows matter as much as the first: the model's probabilities
are **calibrated**, so a customer scored at 0.70 really does churn about
70% of the time. That isn't automatic — see below.

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
3. **Probability calibration**: the tuned pipeline is wrapped in Platt
   scaling (`CalibratedClassifierCV`, `method="sigmoid"`, fit with its own
   internal CV on the training split). See the calibration section below
   for why this is a correctness fix rather than a refinement.
4. **Decision threshold tuning**: the default 0.5 cutoff is rarely optimal
   for an imbalanced target (~27% churn). Using `cross_val_predict` to get
   out-of-fold probabilities on the *training* split only, it sweeps
   thresholds and picks the one maximizing F1 on the "Churn" class —
   without ever looking at the test set, so the threshold isn't overfit to
   it. The tuned value lands at 0.31. (It's tuned on the *calibrated*
   model's out-of-fold probabilities, so the threshold and the served
   probability scale always agree.)

### Calibration: the probabilities now mean what they say

The API returns `churn_probability` and the Streamlit UI renders it as a
percentage. That makes the probability *scale* part of the product's
contract, not just an internal score — and it was wrong.

`scale_pos_weight` buys churn recall by inflating predicted probabilities.
Measuring on the held-out test set showed a systematic upward bias: mean
predicted probability 0.336 against an actual churn rate of 0.265, with
**every** mid-range bin over-predicting by 10–15 points. Customers the UI
labelled "55.2% likely to churn" actually churned 40.5% of the time. This
is a known and documented side effect of class weighting, not a quirk of
this dataset.

Wrapping the model in Platt scaling fixes it, and the classification
metrics come along for free rather than paying for it:

| | ECE | Brier | Churn F1 | Bal. Acc | ROC-AUC |
|---|---|---|---|---|---|
| Uncalibrated | 0.072 | 0.142 | 0.634 | 0.764 | 0.846 |
| **Platt / sigmoid (deployed)** | **0.020** | **0.136** | **0.636** | 0.764 | **0.846** |
| Isotonic | 0.015 | 0.135 | 0.630 | 0.755 | 0.845 |

Calibration error drops by ~72% and every classification metric holds or
improves slightly. Isotonic calibrates marginally better but measurably
costs F1 and balanced accuracy — the usual small-sample overfitting, with
~5,600 training rows — so sigmoid is the better trade here.

Because both methods are monotonic, ranking is untouched, which is why
ROC-AUC barely moves; what changes is that the numbers are now usable for
decisions rather than just ordering. Costs: the artifact grows from 0.14 MB
to 0.71 MB (one fitted model per calibration fold) and prediction latency
rises from ~0.02 ms to ~0.07 ms per row — both irrelevant at this scale.
Training goes from ~2 min to ~2.5 min.

`tests/test_pipeline.py` asserts both that the deployed artifact *is*
calibrated and that reported calibration error stays under 0.04, so a
future retrain can't silently reintroduce the bias.

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

Note these are the numbers *as the comparison was run*, before the blank
`TotalCharges` rows were imputed rather than dropped, and before
calibration. Those changes shifted the split and the probability scale,
which is why the deployed model's current figures at the top of this
section differ slightly. The ranking is what this table is for; it wasn't
re-run for all twelve after each fix.

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

### Feature engineering does not help on this dataset

This is the most useful negative result here, because feature engineering
is the obvious next thing to reach for once model choice is exhausted.

Eight candidate features were tested, drawn from published Telco-churn
repos ([tohid-yousefi](https://github.com/tohid-yousefi/Telco_Customer_Churn_Feature_Engineering),
[DWoyda](https://github.com/DWoyda/telco-customer-churn-ibm-dataset)) plus
two of my own around contract-renewal timing — arguably the strongest
domain hypothesis available, since churn clusters at contract boundaries
and neither `tenure` nor `Contract` alone expresses "about to expire":

| Candidate | CV ROC-AUC | vs baseline |
|---|---|---|
| baseline (current features) | 0.84951 | — |
| `total_services_8` (count over all 8 service columns) | 0.84905 | −0.00046 |
| `contract_cycle` (months into term, cycles done, near-expiry flag) | 0.84905 | −0.00046 |
| `charge_trajectory` (historic avg spend ÷ current rate) | 0.84901 | −0.00050 |
| `no_protection` (missing any of backup/protection/support) | 0.84884 | −0.00067 |
| `charges_per_tenure_ratio` | 0.84852 | −0.00099 |
| `not_engaged_not_senior` | 0.84853 | −0.00098 |
| `avg_service_fee` (MonthlyCharges ÷ services) | 0.84837 | −0.00114 |
| `tenure_year_bucket` (yearly bins) | 0.84810 | −0.00141 |

**Every single one made it worse.** Not by a meaningful amount — all
deltas are an order of magnitude below the fold-to-fold std of 0.012 — but
not one helped.

This isn't a wiring bug: the new columns verifiably reach the model and
get used heavily (`avg_service_fee` draws 9.7% of split importance,
`contract_cycles_done` 4.4%). The model splits on them enthusiastically
and generalizes no better.

So the same question was turned on the features already shipped. Testing
the full engineered set against the **raw 19 columns alone**, paired
across 25 folds (5×5 repeated CV):

```
raw only (19 cols)        0.84946
current (engineered)      0.84894
paired difference         -0.00052   95% CI [-0.00114, +0.00010]   p = 0.118
```

Our own feature engineering doesn't help either. The difference is not
statistically distinguishable from zero, and the point estimate is
slightly *negative*.

The mechanism is the same in both cases: every one of these features —
mine, the repos', and the ones already shipped — is a deterministic
function of columns the model already has. `MonthlyCharges ÷ services`,
`tenure % contract_length`, `TotalCharges ÷ tenure`: a gradient-boosted
tree can already express these by splitting on the inputs. Recombining
existing columns cannot manufacture information; it only adds correlated
copies that dilute split selection.

**Target-derived features, the one class that argument doesn't cover.**
The above all fail for the same structural reason, so it's worth testing
the exception: features built from the *labels* rather than from the
inputs genuinely can add information. Two were tried, both fit inside the
pipeline so per-fold refitting keeps the target statistics away from their
own validation rows. Paired over 15 folds (5×3 repeated CV):

| Configuration | CV ROC-AUC | Paired diff | p |
|---|---|---|---|
| baseline | 0.84898 | — | — |
| + Kaplan-Meier hazard & survival | 0.84693 | **−0.00206** | **0.011** |
| + smoothed target/WoE encoding | 0.84938 | +0.00040 | 0.173 |
| + both | 0.84742 | −0.00156 | 0.037 |

Kaplan-Meier hazard features — motivated by the survival-analysis churn
literature ([hybrid ML + survival framework](https://doi.org/10.3390/info17070680),
and the Kaplan-Meier estimator used in [e-Profits](https://arxiv.org/abs/2507.08860))
— make the model **significantly worse**, not merely no better. The
tenure-conditioned hazard is estimated from ~4,500 training rows spread
over 73 distinct tenure values, so the tail estimates are noisy, and the
booster happily trusts a target-derived feature that is partly memorised
noise.

Target encoding is a wash, for a specific reason: CatBoost's native
categorical handling already computes *ordered* target statistics
internally, and its ordered variant is the leakage-safe version of exactly
this. Adding an explicit encoding duplicates what the model already does,
slightly worse.

**What the literature says actually works.** The one paper found that
reports a large jump —
[churn prediction with social network analysis](https://arxiv.org/pdf/1904.00690),
84% → 93.3% AUC — gets it from call-graph features: who calls whom, and
whether *those* people have already churned. That is not feature
engineering in the sense tested here. It is a different and much richer
data source, on a proprietary operator dataset with call detail records.
It corroborates the conclusion rather than contradicting it: the gain came
from new information, not from recombining the columns already on hand.

The existing features are kept because removing them would churn a
tested, deployed, calibrated model for a change the evidence says is worth
nothing either way. But the finding is the actionable part: on this
dataset, across deterministic recombinations *and* target-derived
features, effort spent on feature engineering is effort wasted. Only
genuinely new *data* — support tickets, usage trends, competitor pricing,
call-graph structure, whether a retention offer was made — can move this.

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

### Bugs and gaps found by auditing the pipeline

**Column-order drift between training and serving.** `app/model.py` and
`train.py` each kept their own hand-maintained copy of the raw feature
order, and they diverged. One-hot pipelines never noticed —
`ColumnTransformer` selects by name, so order is irrelevant — but
CatBoost's native-categorical mode resolves `cat_features` to *positional*
indices at fit time, so a live request could feed a category string into a
slot the model expected to be numeric, crashing `/predict`. Fixed at the
root: both now import a single shared `RAW_FEATURE_ORDER` from
`app/features.py`. `tests/test_pipeline.py` now asserts train/serve
feature parity so this class of drift fails in CI rather than production.

**A train/serve domain mismatch hiding in the data cleaning.** The 11 rows
with blank `TotalCharges` were being dropped. All 11 are `tenure == 0`
customers — they simply hadn't been billed yet, so the true value is
`0.0`, not missing. Dropping them meant the model never saw a brand-new
customer, while the API happily accepted `tenure=0` and extrapolated
silently. Now imputed to `0.0`, with an assertion that fails loudly if a
future data refresh ever puts a blank on a `tenure > 0` row. Net effect on
metrics was mixed and small (ROC-AUC +0.006, balanced accuracy +0.002, F1
−0.002, accuracy −0.009 — and not a clean A/B, since 11 extra rows change
the train/test split itself); the reason to do it is correctness, not the
scoreboard.

**Unbounded numeric inputs.** `MonthlyCharges` and `TotalCharges` had no
upper bound and `tenure` allowed up to 100, against training ranges of
18.25–118.75, 0–8684.80 and 0–72. A gradient-boosted tree doesn't
extrapolate — it clamps to the nearest leaf — so `MonthlyCharges: 99999`
returned a confident-looking 0.70 rather than an error. Bounds are now set
with deliberate headroom above the observed range, and out-of-range input
gets a 422 instead of a fabricated answer.

**A container that could come up broken.** `start.sh` waited 30s for the
API and then started the UI *regardless*. If the backend failed, the Space
would look healthy while erroring on every prediction — silent degradation
is harder to diagnose than a crash. It now aborts if the API dies or never
becomes healthy, and the Dockerfile has a `HEALTHCHECK` that probes the
API rather than the UI.

**The Dockerfile ran as root.** Hugging Face Spaces runs Docker containers
as UID 1000, and Streamlit needs a writable `HOME` for its config/cache —
a root-owned `/root` works locally and fails there. Now builds and runs as
a non-root `appuser` with a real home directory.

**Nothing ever built the image.** The deployment target is a Docker image,
but it had never been built once — no local Docker, no CI. That was the
single largest deploy risk. `.github/workflows/ci.yml` now runs the test
suite and, separately, builds the image, starts the container, and asserts
that `/health`, `/predict` and the Streamlit UI all actually respond
before anything reaches a Space.

### Feature importance

Full ranked list is written to `model/feature_importance.json` on every
training run, taken from CatBoost's own `feature_importances_`. Because
native mode doesn't one-hot encode, these are importances per *raw
column*, not per category level. Top 5:

| Feature | Importance |
|---|---|
| `Contract` | 23.7% |
| `InternetService` | 14.8% |
| `tenure` | 10.1% |
| `MonthlyCharges` | 6.5% |
| `PaymentMethod` | 5.1% |

...and the tail — the 10 least useful of the 29 raw + engineered columns:

| Feature | Importance |
|---|---|
| `TechSupport` | 1.12% |
| `StreamingTV` | 1.07% |
| `SeniorCitizen` | 0.69% |
| `Dependents` | 0.67% |
| `has_streaming` | 0.60% |
| `gender` | 0.50% |
| `OnlineBackup` | 0.38% |
| `manual_payment` | 0.27% |
| `DeviceProtection` | 0.12% |
| `Partner` | 0.02% (effectively unused, as in every earlier run) |

Values are averaged across the five calibration folds, each of which holds
its own fitted CatBoost. Only 4 of 29 columns fall below the 0.5%-each
dead-weight threshold, carrying 0.8% between them — much cleaner than the
one-hot-expanded counts earlier in this project, though those were never
comparing like for like (one raw column vs. several one-hot levels of it).
`Contract` and `InternetService` alone account for 38% of total
importance, consistent with every other lens applied to this dataset —
segment analysis, permutation importance, and the reference paper's SHAP
results all agree these two (plus tenure) carry most of the real signal.

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
the tuned threshold rather than fixed 0.33/0.66 cutoffs, so `Medium`
always straddles the actual Yes/No decision boundary. That matters here
because the threshold has ranged from 0.26 to 0.63 across models and
moved again (0.44 → 0.31) when calibration rescaled the probabilities;
hardcoded bands would have silently desynced from the decision boundary.

### Not done: profit-driven thresholds

The threshold is tuned for F1, which treats a missed churner and a wasted
retention offer as equally costly. They usually aren't. Current work in
this area ([e-Profits](https://arxiv.org/abs/2507.08860)) selects the
threshold — and even the model — by expected profit, using customer
lifetime value, intervention cost, and tenure-conditioned retention
probability.

That's the right next step, and it's deliberately *not* implemented here:
it needs real CLV and campaign-cost figures, and inventing them would
produce something that looks rigorous while being arbitrary. The
groundwork is done, though — cost-based thresholding requires calibrated
probabilities to be meaningful, and those now exist. With real numbers,
swapping `best_threshold_for_f1` for an expected-cost objective is a small
change to one function.

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Train the model (writes model/churn_model.joblib, decision_threshold.json,
# metrics.json, feature_importance.json, feature_schema.json). Runs a
# 20-iteration hyperparameter search with 5-fold CV, plus calibration
# (which fits one model per fold), so it takes ~2.5min.
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
  "churn_probability": 0.6947,
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

This repo pushes straight to a Space with no changes. The YAML frontmatter
at the top of this file (`sdk: docker`, `app_port: 7860`) is what Spaces
reads to configure the build, and the image already runs as UID 1000 with
a writable `HOME`, which is what Spaces requires.

**1. Create the Space.** Go to https://huggingface.co/new-space and pick:

| Setting | Value |
|---|---|
| Space name | e.g. `telco-churn-app` |
| License | your choice |
| SDK | **Docker** → *Blank* |
| Hardware | *CPU basic* (free) is enough |
| Visibility | Public or Private |

**2. Authenticate git against Hugging Face.** Create a token with **write**
scope at https://huggingface.co/settings/tokens, then:

```bash
pip install -U huggingface_hub
huggingface-cli login          # paste the write token
```

**3. Push this repo to the Space:**

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
git push hf main
```

**4. Watch it build.** The Space page shows build logs. First build takes
a few minutes (it installs CatBoost, pandas and Streamlit). When it goes
green, the Streamlit UI is at
`https://huggingface.co/spaces/<your-username>/<space-name>`.

**5. Confirm what actually deployed:**

```bash
curl https://<your-username>-<space-name>.hf.space/model-info
```

Notes and gotchas:

- **Only port 7860 is exposed publicly.** Spaces routes to `app_port`, so
  the Streamlit UI is reachable from outside and the FastAPI backend on
  8000 is internal to the container. That's intentional — Streamlit calls
  it over localhost. To expose the API publicly instead, set
  `app_port: 8000` in the frontmatter and change `start.sh` to run uvicorn
  in the foreground.
- **The model is committed to the repo** (`model/churn_model.joblib`,
  0.7 MB), so the Space needs no training step and no external storage.
- **Free Spaces sleep when idle** and take ~30s to wake. The container
  itself starts in about 4 seconds once running.
- **If the build fails**, compare against CI: the `docker` job here builds
  the identical image and boots it, so a green CI run and a failing Space
  points at a platform difference rather than the Dockerfile.

## Deploying to Render instead

Render builds the same `Dockerfile` with no changes (New → Web Service →
point at this repo → Environment: Docker). Render injects `$PORT` and
expects the service to listen on it; `start.sh` already reads `$PORT` for
the Streamlit UI (falling back to 7860 for Hugging Face Spaces), so the
image works on either platform as-is.

The FastAPI backend stays on `$API_PORT` (default 8000) inside the
container. To deploy the API alone without the UI, override the command:

```dockerfile
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(also mirrored by IBM as sample data for the same dataset) — 7,043 customers,
19 features (demographics, account info, services subscribed), binary churn
label.
