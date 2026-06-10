import json
import numpy as np
from pathlib import Path
from typing import Optional


FEATURE_COLS = [
    "career_arc_score",
    "product_ratio",
    "product_industry_score",
    "is_ic",
    "tenure_score",
    "production_score",
    "pre_llm_score",
    "yoe_fit",
    "normalized_skill_score",
    "relevant_skill_months_score",
    "must_have_count",
    "gh_score",
    "saved_score",
    "search_score",
    "salary_fit",
    "availability_score",
    "responsiveness_score",
    "notice_score",
    "engagement_score",
    "recency_score",
    "location_score",
    "is_tier1_india",
    "is_india",
    "will_relocate",
]


def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([feat.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32)


def generate_pseudo_labels(
    candidate_ids: list[str],
    features: dict[str, dict],
    semantic_scores: dict[str, float],
) -> np.ndarray:
    """
    Build pseudo-relevance labels [0.0 - 3.0] for XGBoost training.

    Label logic:
    3.0 = strong fit  (high semantic + high production score + product company)
    2.0 = good fit    (decent semantic + some production evidence)
    1.0 = weak fit    (passes filter but limited ML signal)
    0.0 = poor fit    (barely passed filter, low semantic score)
    """
    labels = []
    for cid in candidate_ids:
        feat = features.get(cid, {})
        sem = semantic_scores.get(cid, 0.0)

        production = feat.get("production_score", 0)
        product_ratio = feat.get("product_ratio", 0)
        skill_score = feat.get("normalized_skill_score", 0)
        pre_llm = feat.get("pre_llm_score", 0)
        avail = feat.get("availability_score", 0)

        # Composite relevance signal
        relevance = (
            0.35 * sem
            + 0.25 * production
            + 0.15 * product_ratio
            + 0.15 * skill_score
            + 0.10 * pre_llm
        )

        # Availability modifier — a great candidate who's unreachable is less relevant
        relevance = relevance * (0.6 + 0.4 * avail)

        # Convert to 0-3 scale
        if relevance >= 0.65:
            label = 3.0
        elif relevance >= 0.45:
            label = 2.0
        elif relevance >= 0.28:
            label = 1.0
        else:
            label = 0.0

        labels.append(label)

    return np.array(labels, dtype=np.float32)


def train_ltr_model(
    candidate_ids: list[str],
    features: dict[str, dict],
    semantic_scores: dict[str, float],
    model_path: Path,
) -> None:
    import xgboost as xgb

    X = np.array([features_to_vector(features.get(cid, {})) for cid in candidate_ids])
    y = generate_pseudo_labels(candidate_ids, features, semantic_scores)

    # Group array for LambdaMART — treat all candidates as one query group
    groups = np.array([len(candidate_ids)])

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        verbose_eval=False,
    )

    model.save_model(str(model_path))
    print(f"[ltr] Model trained on {len(candidate_ids):,} candidates, saved to {model_path}")

    # Log feature importances
    importance = model.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
    print("[ltr] Top features by gain:")
    for fname, score in top_features:
        idx = int(fname[1:]) if fname.startswith("f") else 0
        col = FEATURE_COLS[idx] if idx < len(FEATURE_COLS) else fname
        print(f"      {col}: {score:.1f}")


def load_ltr_model(model_path: Path) -> Optional[object]:
    if not model_path.exists():
        return None
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(model_path))
        return model
    except Exception as e:
        print(f"[ltr] Warning: could not load model: {e}")
        return None


def ltr_score(
    candidate_ids: list[str],
    features: dict[str, dict],
    model,
) -> dict[str, float]:
    import xgboost as xgb

    X = np.array([features_to_vector(features.get(cid, {})) for cid in candidate_ids])
    dmat = xgb.DMatrix(X)
    raw_scores = model.predict(dmat)

    # Normalize to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        normalized = (raw_scores - s_min) / (s_max - s_min)
    else:
        normalized = np.ones_like(raw_scores) * 0.5

    return {cid: float(normalized[i]) for i, cid in enumerate(candidate_ids)}
