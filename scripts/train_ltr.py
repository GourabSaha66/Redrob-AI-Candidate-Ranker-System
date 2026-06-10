import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


FEATURE_COLS = [
    "career_arc_score",
    "product_ratio",
    "is_ic",
    "tenure_score",
    "production_score",
    "pre_llm_score",
    "yoe_fit",
    "assessment_credibility",
    "gh_score",
    "saved_score",
    "views_score",
    "salary_fit",
    "skills_score",
    "must_have_count",
    "availability_score",
    "responsiveness_score",
    "notice_score",
    "engagement_score",
    "recency_score",
    "behavioral_multiplier",
    "location_score",
    "is_tier1_india",
    "is_india",
    "will_relocate",
]


def build_feature_matrix(ids, features):
    X = []
    for cid in ids:
        feat = features.get(cid, {})
        row = [float(feat.get(col, 0.0)) for col in FEATURE_COLS]
        X.append(row)
    return np.array(X, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.artifacts)

    print("[ltr] Loading artifacts ...")
    with open(out_dir / "candidate_ids.json") as f:
        ids = json.load(f)
    with open(out_dir / "candidate_features.json") as f:
        features = json.load(f)

    embeddings = np.load(str(out_dir / "candidate_embeddings.npy"))

    from config import JD_TECHNICAL_REQUIREMENTS, JD_CAREER_NARRATIVE
    from sentence_transformers import SentenceTransformer

    print("[ltr] Computing JD similarity scores (pseudo-labels) ...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    jd_embs = model.encode(
        [JD_TECHNICAL_REQUIREMENTS.strip(), JD_CAREER_NARRATIVE.strip()],
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    jd_vec = jd_embs.mean(axis=0)
    jd_vec = jd_vec / np.linalg.norm(jd_vec)
    bienc_scores = embeddings @ jd_vec

    # Combine bienc score with career features for a richer pseudo-label
    # This is more robust than bienc alone
    prod_scores = np.array([features[i].get("production_score", 0) for i in ids])
    product_ratios = np.array([features[i].get("product_ratio", 0) for i in ids])
    must_haves = np.array([min(1.0, features[i].get("must_have_count", 0) / 5.0) for i in ids])

    # Pseudo-label: weighted combination of semantic + structured signals
    pseudo_labels = (
        0.50 * bienc_scores +
        0.20 * prod_scores +
        0.15 * product_ratios +
        0.15 * must_haves
    ).astype(np.float32)

    print(f"[ltr] Building feature matrix for {len(ids):,} candidates ...")
    X = build_feature_matrix(ids, features)
    y = pseudo_labels

    print(f"[ltr] Feature matrix: {X.shape}, labels range: [{y.min():.3f}, {y.max():.3f}]")

    try:
        import xgboost as xgb
    except ImportError:
        print("[ltr] Installing xgboost ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "--break-system-packages", "-q"])
        import xgboost as xgb

    print("[ltr] Training XGBoost ranker ...")

    # Use rank:pairwise objective — designed for learning-to-rank
    # Group all candidates as one query (ranking against the same JD)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
    dtrain.set_group([len(ids)])

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 300,
        "tree_method": "hist",
        "seed": 42,
    }

    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        verbose_eval=50,
    )

    # Save model
    model_path = out_dir / "ltr_model.xgb"
    model_xgb.save_model(str(model_path))
    print(f"[ltr] Model saved to {model_path}")

    # Show feature importance
    importance = model_xgb.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n[ltr] Top feature importances (gain):")
    for feat, score in sorted_imp[:12]:
        bar = "█" * int(score / sorted_imp[0][1] * 30)
        print(f"  {feat:<30} {score:>8.1f}  {bar}")

    # Generate and save LTR scores for all candidates
    ltr_scores = model_xgb.predict(dtrain)
    ltr_score_map = {cid: float(ltr_scores[i]) for i, cid in enumerate(ids)}
    with open(out_dir / "ltr_scores.json", "w") as f:
        json.dump(ltr_score_map, f)
    print(f"\n[ltr] LTR scores saved to {out_dir}/ltr_scores.json")
    print("[ltr] Done. rank.py will automatically use these scores.")


if __name__ == "__main__":
    main()
