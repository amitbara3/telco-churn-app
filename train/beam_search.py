"""Beam search feature selection over the Telco feature set.

Runs entirely on the training split (the held-out test set is never
touched during search — only used afterward, once, to validate whatever
the search picks). Candidates are the "logical" columns FeatureEngineer
produces (raw + engineered), not individual one-hot output columns —
selection happens at the column level so a categorical feature's one-hot
levels are always included/excluded as a unit.

Algorithm (forward beam search):
  - Start from the empty feature set.
  - Each round, expand every subset currently in the beam by adding one
    more not-yet-included feature (tried in order of aggregate importance,
    from model/feature_importance.json, as a search-order heuristic only —
    it doesn't restrict which features can be added).
  - Score every candidate subset by 5-fold StratifiedKFold cross-validated
    balanced accuracy on the training split, using the already-tuned
    LightGBM hyperparameters as a fixed, fast evaluator.
  - Keep the top BEAM_WIDTH candidates as the next round's beam.
  - Track the best subset seen across all rounds. Stop once that best
    score hasn't improved for more than PATIENCE consecutive rounds (or
    once every feature has been added).

Usage:
    python train/beam_search.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, FeatureEngineer  # noqa: E402
from train.train import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    MODEL_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    load_data,
)

BEAM_WIDTH = 5
PATIENCE = 10

IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
BEAM_RESULT_PATH = MODEL_DIR / "beam_search_result.json"

ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_CANDIDATES = ALL_CATEGORICAL + ALL_NUMERIC


def load_best_lightgbm_params() -> dict:
    metrics = json.loads(METRICS_PATH.read_text())
    best_params = metrics["candidates"]["lightgbm"]["best_params"]
    return {k.replace("model__", ""): v for k, v in best_params.items()}


def importance_order() -> list[str]:
    """Aggregates the persisted one-hot feature importances (imp tail
    analysis) back up to raw column names, to order candidate expansion —
    try the most-promising not-yet-included feature first each round."""
    rows = json.loads(IMPORTANCE_PATH.read_text())
    agg = dict.fromkeys(ALL_CANDIDATES, 0.0)
    for row in rows:
        name, value = row["feature"], row["importance"]
        if name.startswith("num__"):
            raw = name[len("num__"):]
        else:
            stripped = name[len("cat__"):]
            matches = [c for c in ALL_CATEGORICAL if stripped == c or stripped.startswith(c + "_")]
            raw = max(matches, key=len)
        agg[raw] += value
    return sorted(agg, key=agg.get, reverse=True)


def build_subset_pipeline(cat_cols: list[str], num_cols: list[str], params: dict) -> Pipeline:
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    ct = ColumnTransformer(transformers=transformers).set_output(transform="pandas")
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, subsample_freq=1, verbosity=-1, **params)
    return Pipeline(steps=[("engineer", FeatureEngineer()), ("preprocess", ct), ("model", model)])


def score_subset(subset: frozenset, X_train, y_train, cv, params: dict) -> float:
    cat_cols = [c for c in subset if c in ALL_CATEGORICAL]
    num_cols = [c for c in subset if c in ALL_NUMERIC]
    pipeline = build_subset_pipeline(cat_cols, num_cols, params)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=1)
    return float(scores.mean())


def beam_search(X_train, y_train, params: dict) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    order = importance_order()

    cache: dict[frozenset, float] = {}
    beam = [frozenset()]
    best_overall_score, best_overall_subset = -np.inf, frozenset()
    rounds_without_improvement = 0
    history = []

    for round_num in range(1, len(order) + 1):
        candidates = set()
        for subset in beam:
            for f in order:
                if f not in subset:
                    candidates.add(subset | {f})

        if not candidates:
            break

        new_candidates = [c for c in candidates if c not in cache]
        if new_candidates:
            scored = Parallel(n_jobs=-1)(
                delayed(score_subset)(c, X_train, y_train, cv, params) for c in new_candidates
            )
            cache.update(zip(new_candidates, scored))

        ranked = sorted(candidates, key=lambda c: cache[c], reverse=True)
        beam = ranked[:BEAM_WIDTH]

        round_best_subset = ranked[0]
        round_best_score = cache[round_best_subset]
        history.append(
            {
                "round": round_num,
                "best_subset": sorted(round_best_subset),
                "best_balanced_accuracy": round_best_score,
            }
        )
        print(
            f"Round {round_num}: best bal_acc={round_best_score:.4f}  "
            f"size={len(round_best_subset)}  features={sorted(round_best_subset)}"
        )

        if round_best_score > best_overall_score:
            best_overall_score, best_overall_subset = round_best_score, round_best_subset
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement > PATIENCE:
                print(f"Stopping: balanced accuracy hasn't improved in {PATIENCE} rounds.")
                break

    return {
        "best_subset": sorted(best_overall_subset),
        "best_balanced_accuracy": best_overall_score,
        "beam_width": BEAM_WIDTH,
        "patience": PATIENCE,
        "history": history,
    }


def main() -> None:
    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    params = load_best_lightgbm_params()
    result = beam_search(X_train, y_train, params)

    print(
        f"\nBest subset found ({len(result['best_subset'])} of {len(ALL_CANDIDATES)} "
        f"features, train-CV bal_acc={result['best_balanced_accuracy']:.4f}):"
    )
    print(result["best_subset"])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    full_score = score_subset(frozenset(ALL_CANDIDATES), X_train, y_train, cv, params)
    print(f"\nFull feature set ({len(ALL_CANDIDATES)} features) train-CV bal_acc={full_score:.4f}")
    result["full_feature_set_balanced_accuracy"] = full_score

    BEAM_RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Saved beam search result to {BEAM_RESULT_PATH}")


if __name__ == "__main__":
    main()
