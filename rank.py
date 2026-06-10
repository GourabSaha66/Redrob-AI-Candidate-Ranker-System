import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    TOP_K_BIENCODER, TOP_K_CROSSENCODER, FINAL_TOP_N,
    WEIGHT_ROLE_FIT, WEIGHT_CAREER_ARC, WEIGHT_AVAILABILITY, WEIGHT_LOCATION,
    BIENC_WEIGHT, CROSSENC_WEIGHT,
    JD_TECHNICAL_REQUIREMENTS, JD_CAREER_NARRATIVE,
    BIENCODER_MODEL, CROSSENCODER_MODEL,
)
from reasoning import generate_reasoning


def load_artifacts(artifacts_dir: Path):
    print("[rank] Loading artifacts ...")
    embeddings = np.load(str(artifacts_dir / "candidate_embeddings.npy"))
    with open(artifacts_dir / "candidate_ids.json") as f:
        candidate_ids = json.load(f)
    with open(artifacts_dir / "candidate_features.json") as f:
        features = json.load(f)
    with open(artifacts_dir / "candidate_slim.json") as f:
        slim = json.load(f)

    reasoning_cache = {}
    cache_path = artifacts_dir / "reasoning_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            reasoning_cache = json.load(f)
        print(f"[rank] Loaded {len(reasoning_cache)} pre-generated reasonings from cache")
    else:
        print("[rank] No reasoning cache found — will use template reasoning")

    print(f"[rank] {len(candidate_ids):,} candidates, embeddings {embeddings.shape}")
    return embeddings, candidate_ids, features, slim, reasoning_cache


def biencoder_retrieve(embeddings, candidate_ids, model, top_k):
    jd_embs = model.encode(
        [JD_TECHNICAL_REQUIREMENTS.strip(), JD_CAREER_NARRATIVE.strip()],
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    jd_vec = jd_embs.mean(axis=0)
    jd_vec = jd_vec / np.linalg.norm(jd_vec)
    scores = embeddings @ jd_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(candidate_ids[i], float(scores[i]), int(i)) for i in top_indices]
    print(f"[bienc] Top-{top_k} retrieved. Range: {results[-1][1]:.3f} – {results[0][1]:.3f}")
    return results


def crossencoder_rerank(top_candidates, slim, cross_encoder, top_k):
    jd_query = (JD_TECHNICAL_REQUIREMENTS + " " + JD_CAREER_NARRATIVE).strip()[:1000]
    pairs = []
    for cid, _, _ in top_candidates:
        s = slim.get(cid, {})
        text = (
            f"{s.get('current_title','')} at {s.get('current_company','')} "
            f"({s.get('years_of_experience',0)}yr). "
            f"Skills: {', '.join(s.get('skills',[])[:5])}. "
            f"Location: {s.get('location','')} {s.get('country','')}"
        )
        pairs.append([jd_query, text])

    print(f"[crossenc] Scoring {len(pairs)} pairs ...")
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=True)
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    ce_norm = (ce_scores - ce_min) / (ce_max - ce_min) if ce_max > ce_min else np.ones_like(ce_scores) * 0.5

    results = [
        (top_candidates[i][0], top_candidates[i][1], float(ce_norm[i]))
        for i in range(len(top_candidates))
    ]
    results.sort(key=lambda x: BIENC_WEIGHT * x[1] + CROSSENC_WEIGHT * x[2], reverse=True)
    return results[:top_k]


def compute_final_scores(reranked, features, ltr_model):
    from ltr import ltr_score

    candidate_ids = [r[0] for r in reranked]

    if ltr_model is not None:
        ltr_scores = ltr_score(candidate_ids, features, ltr_model)
        ltr_available = True
    else:
        ltr_scores = {cid: 0.5 for cid in candidate_ids}
        ltr_available = False

    results = []
    for cid, bienc_score, ce_score in reranked:
        feat = features.get(cid, {})

        semantic = BIENC_WEIGHT * bienc_score + CROSSENC_WEIGHT * ce_score
        ltr = ltr_scores.get(cid, 0.5)

        role_signal = (0.40 * semantic + 0.60 * ltr) if ltr_available else semantic

        role_fit = WEIGHT_ROLE_FIT * role_signal + WEIGHT_CAREER_ARC * feat.get("career_arc_score", 0.5)
        avail    = WEIGHT_AVAILABILITY * feat.get("availability_score", 0.5)
        loc      = WEIGHT_LOCATION * feat.get("location_score", 0.5)

        pre_final = role_fit + avail + loc
        final     = pre_final * feat.get("behavioral_multiplier", 0.75)

        results.append((cid, round(final, 6), feat, semantic, ltr))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def load_full_records(candidates_path: Path, needed_ids: set) -> dict:
    print(f"[rank] Loading full records for top-{len(needed_ids)} ...")
    found = {}
    opener = (gzip.open(candidates_path, "rt", encoding="utf-8")
              if candidates_path.suffix == ".gz"
              else open(candidates_path, encoding="utf-8"))
    with opener as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in needed_ids:
                found[c["candidate_id"]] = c
                if len(found) == len(needed_ids):
                    break
    return found


def main():
    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    artifacts_dir   = Path(args.artifacts)
    out_path        = Path(args.out)

    if not artifacts_dir.exists():
        print("[ERROR] artifacts/ not found. Run scripts/precompute_embeddings.py first.")
        sys.exit(1)

    embeddings, candidate_ids, features, slim, reasoning_cache = load_artifacts(artifacts_dir)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("[rank] Loading models ...")
    bienc    = SentenceTransformer(BIENCODER_MODEL)
    crossenc = CrossEncoder(CROSSENCODER_MODEL)

    from ltr import load_ltr_model
    ltr_model = load_ltr_model(artifacts_dir / "ltr_model.ubj")
    print(f"[rank] LTR model: {'loaded ✓' if ltr_model else 'not found — semantic only'}")

    t1 = time.time()
    top500 = biencoder_retrieve(embeddings, candidate_ids, bienc, TOP_K_BIENCODER)
    print(f"[rank] Bi-encoder: {time.time()-t1:.1f}s")

    t2 = time.time()
    top200 = crossencoder_rerank(top500, slim, crossenc, TOP_K_CROSSENCODER)
    print(f"[rank] Cross-encoder: {time.time()-t2:.1f}s")

    scored  = compute_final_scores(top200, features, ltr_model)
    top100  = scored[:FINAL_TOP_N]

    full_records = load_full_records(candidates_path, {row[0] for row in top100})

    rows = []
    cache_hits = 0
    for rank_pos, (cid, score, feat, semantic, ltr) in enumerate(top100, 1):
        candidate = full_records.get(cid, {
            "candidate_id": cid,
            "profile": slim.get(cid, {}),
            "career_history": [],
            "skills": [],
            "redrob_signals": {},
        })
        reasoning = generate_reasoning(candidate, feat, semantic, score, rank_pos, cache=reasoning_cache)
        if cid in reasoning_cache:
            cache_hits += 1
        rows.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        score,
            "reasoning":    reasoning,
        })

    print(f"[rank] Reasoning: {cache_hits}/100 from LLM cache, {100-cache_hits} from template")

    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow([row["candidate_id"], row["rank"], f"{row['score']:.6f}", row["reasoning"]])

    total = time.time() - t_start
    print(f"\n[done] {total:.1f}s {'✓ within 5min' if total < 300 else '✗ exceeded 5min'}")
    print(f"[done] Written to {out_path}")
    print(f"[sanity] rows={len(rows)} | monotone={all(rows[i]['score']>=rows[i+1]['score'] for i in range(len(rows)-1))} | unique={len({r['candidate_id'] for r in rows})}")


if __name__ == "__main__":
    main()
