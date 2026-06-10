import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_candidates(path: str) -> list:
    p = Path(path)
    print(f"[load] Reading {p} ...")
    candidates = []
    opener = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"[load] Loaded {len(candidates):,} candidates")
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-reasoning", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path  = out_dir / "candidate_embeddings.npy"
    ids_path  = out_dir / "candidate_ids.json"
    feat_path = out_dir / "candidate_features.json"
    slim_path = out_dir / "candidate_slim.json"
    ltr_path  = out_dir / "ltr_model.ubj"

    # ── Step 1: Filter ────────────────────────────────────────────────────
    candidates = load_candidates(args.candidates)
    from filters import apply_filters
    kept, excluded = apply_filters(candidates)
    with open(out_dir / "excluded_candidates.json", "w") as f:
        json.dump(excluded, f, indent=2)

    # ── Step 2: Embeddings (skip if exists) ───────────────────────────────
    if emb_path.exists() and ids_path.exists():
        print(f"[embed] Already exists — skipping. Delete {emb_path} to redo.")
        embeddings = np.load(str(emb_path))
        with open(ids_path) as f:
            ids = json.load(f)
    else:
        from text_builder import build_candidate_text
        from sentence_transformers import SentenceTransformer
        print("[text] Building text representations ...")
        texts = [build_candidate_text(c, mode="full") for c in kept]
        print(f"[model] Loading {args.model} ...")
        model = SentenceTransformer(args.model)
        print(f"[embed] Embedding {len(texts):,} candidates ...")
        t0 = time.time()
        embeddings = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        print(f"[embed] Done in {time.time()-t0:.1f}s. Shape: {embeddings.shape}")
        np.save(str(emb_path), embeddings.astype(np.float32))
        ids = [c["candidate_id"] for c in kept]
        with open(ids_path, "w") as f:
            json.dump(ids, f)

    # ── Step 3: Features (skip if exists) ────────────────────────────────
    if feat_path.exists():
        print(f"[features] Already exists — skipping.")
        with open(feat_path) as f:
            feature_map = json.load(f)
    else:
        from features import featurize
        print("[features] Computing features ...")
        feature_map = {c["candidate_id"]: featurize(c) for c in kept}
        with open(feat_path, "w") as f:
            json.dump(feature_map, f)

    # ── Step 4: LTR model (skip if exists) ───────────────────────────────
    if ltr_path.exists():
        print(f"[ltr] Already exists — skipping.")
    else:
        from sentence_transformers import SentenceTransformer
        from config import JD_TECHNICAL_REQUIREMENTS, JD_CAREER_NARRATIVE
        from ltr import train_ltr_model
        print("[ltr] Computing JD similarity for pseudo-labels ...")
        model = SentenceTransformer(args.model)
        jd_embs = model.encode(
            [JD_TECHNICAL_REQUIREMENTS.strip(), JD_CAREER_NARRATIVE.strip()],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        jd_vec = jd_embs.mean(axis=0)
        jd_vec = jd_vec / np.linalg.norm(jd_vec)
        raw_scores = embeddings @ jd_vec
        s_min, s_max = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones_like(raw_scores) * 0.5
        semantic_scores = {ids[i]: float(norm_scores[i]) for i in range(len(ids))}
        print("[ltr] Training XGBoost LambdaMART ...")
        train_ltr_model(ids, feature_map, semantic_scores, ltr_path)

    # ── Step 5: Slim store (skip if exists) ───────────────────────────────
    if slim_path.exists():
        print(f"[slim] Already exists — skipping.")
    else:
        slim = {}
        for c in kept:
            p = c["profile"]
            s = c.get("redrob_signals", {})
            slim[c["candidate_id"]] = {
                "current_title":           p.get("current_title", ""),
                "current_company":         p.get("current_company", ""),
                "years_of_experience":     p.get("years_of_experience", 0),
                "location":                p.get("location", ""),
                "country":                 p.get("country", ""),
                "skills":                  [sk["name"] for sk in c.get("skills", [])
                                            if sk.get("proficiency") in ("advanced", "expert")][:6],
                "last_active_date":        s.get("last_active_date", ""),
                "notice_period_days":      s.get("notice_period_days", 90),
                "recruiter_response_rate": s.get("recruiter_response_rate", 0),
                "open_to_work_flag":       s.get("open_to_work_flag", False),
            }
        with open(slim_path, "w") as f:
            json.dump(slim, f)

    # ── Step 6: LLM Reasoning ─────────────────────────────────────────────
    if args.skip_reasoning:
        print("\n[reasoning] Skipped (--skip-reasoning flag).")
    else:
        cache_path = out_dir / "reasoning_cache.json"
        existing = {}
        if cache_path.exists():
            with open(cache_path) as f:
                existing = json.load(f)
            print(f"[reasoning] Cache has {len(existing)} entries already.")

        if len(existing) >= 300:
            print("[reasoning] Already have 300+ reasonings — skipping.")
        else:
            print("\n[reasoning] Pre-generating LLM reasoning for top-300 candidates ...")
            combined = {
                cid: 0.6 * float(np.dot(embeddings[ids.index(cid)], embeddings[ids.index(cid)]))
                     + 0.4 * feature_map.get(cid, {}).get("career_arc_score", 0)
                for cid in ids[:500]
            }
            # Simpler: just use career arc + feature score to pick top 300
            scored = sorted(
                [(cid, feature_map.get(cid, {}).get("career_arc_score", 0) +
                       feature_map.get(cid, {}).get("production_score", 0))
                 for cid in ids],
                key=lambda x: x[1], reverse=True
            )
            top300_ids = [cid for cid, _ in scored[:300]]
            kept_map = {c["candidate_id"]: c for c in kept}

            candidates_data = [
                (cid, kept_map[cid], feature_map.get(cid, {}), 0.0, rank + 1)
                for rank, cid in enumerate(top300_ids)
                if cid in kept_map and cid not in existing
            ]

            if not candidates_data:
                print("[reasoning] All top-300 already in cache.")
            else:
                from reasoning import generate_llm_reasoning_batch
                generate_llm_reasoning_batch(candidates_data, cache_path)

    print(f"\n[done] All artifacts saved to {out_dir}/")
    print("       Now run: python rank.py --candidates data/candidates.jsonl --out submission.csv")


if __name__ == "__main__":
    main()
