import json
import sys
import time
import csv
import io
from pathlib import Path

import streamlit as st
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="Redrob AI Candidate Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Redrob AI Candidate Ranker")
st.caption("Senior AI Engineer — Founding Team | Hybrid Semantic + Behavioral Ranking")


@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    return SentenceTransformer("BAAI/bge-small-en-v1.5"), CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_data
def get_jd_vector(_bienc):
    from config import JD_TECHNICAL_REQUIREMENTS, JD_CAREER_NARRATIVE
    embs = _bienc.encode([JD_TECHNICAL_REQUIREMENTS.strip(), JD_CAREER_NARRATIVE.strip()], normalize_embeddings=True, convert_to_numpy=True)
    v = embs.mean(axis=0)
    return v / np.linalg.norm(v)


st.sidebar.header("Input")
input_mode = st.sidebar.radio("Input mode", ["Upload JSON file", "Paste JSON"])

candidates = None
if input_mode == "Upload JSON file":
    uploaded = st.sidebar.file_uploader("Upload candidates JSON (≤100)", type=["json"])
    if uploaded:
        try:
            candidates = json.load(uploaded)
            st.sidebar.success(f"Loaded {len(candidates)} candidates")
        except Exception as e:
            st.sidebar.error(f"Parse error: {e}")
else:
    text = st.sidebar.text_area("Paste JSON array", height=200)
    if text.strip():
        try:
            candidates = json.loads(text)
            st.sidebar.success(f"Loaded {len(candidates)} candidates")
        except Exception as e:
            st.sidebar.error(f"Parse error: {e}")

top_n = st.sidebar.slider("Top N to display", 10, 100, 100)

if candidates is None:
    st.info("👈 Upload sample_candidates.json to get started")
    st.stop()

if len(candidates) > 100:
    st.warning(f"Demo limited to 100 candidates. Truncating.")
    candidates = candidates[:100]

if st.button("🚀 Run Ranker", type="primary"):
    t0 = time.time()

    with st.spinner("Loading models ..."):
        bienc, crossenc = load_models()
        jd_vec = get_jd_vector(bienc)

    from filters import apply_filters
    kept, excluded = apply_filters(candidates)
    col1, col2, col3 = st.columns(3)
    col1.metric("After filter", len(kept), f"-{len(excluded)}")

    if not kept:
        st.error("All candidates filtered. Check your input.")
        st.stop()

    from text_builder import build_candidate_text
    texts = [build_candidate_text(c, mode="full") for c in kept]
    embeddings = bienc.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    bienc_scores = embeddings @ jd_vec

    top_k = min(50, len(kept))
    top_indices = np.argsort(bienc_scores)[::-1][:top_k]

    from config import JD_TECHNICAL_REQUIREMENTS, JD_CAREER_NARRATIVE
    jd_query = (JD_TECHNICAL_REQUIREMENTS + " " + JD_CAREER_NARRATIVE).strip()[:1000]
    pairs = []
    for idx in top_indices:
        c = kept[idx]
        p = c.get("profile", {})
        skills = [s["name"] for s in c.get("skills", []) if s.get("proficiency") == "advanced"][:5]
        pairs.append([jd_query, f"{p.get('current_title','')} at {p.get('current_company','')} ({p.get('years_of_experience',0)}yr). Skills: {', '.join(skills)}"])

    ce_scores = crossenc.predict(pairs, show_progress_bar=False)
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    if ce_max > ce_min:
        ce_scores = (ce_scores - ce_min) / (ce_max - ce_min)

    from features import featurize
    from config import BIENC_WEIGHT, CROSSENC_WEIGHT, WEIGHT_ROLE_FIT, WEIGHT_CAREER_ARC, WEIGHT_AVAILABILITY, WEIGHT_LOCATION

    final_results = []
    for j, idx in enumerate(top_indices):
        c = kept[idx]
        feat = featurize(c)
        semantic = BIENC_WEIGHT * float(bienc_scores[idx]) + CROSSENC_WEIGHT * float(ce_scores[j])
        role_fit = WEIGHT_ROLE_FIT * semantic + WEIGHT_CAREER_ARC * feat.get("career_arc_score", 0.5)
        total = role_fit + WEIGHT_AVAILABILITY * feat.get("availability_score", 0.5) + WEIGHT_LOCATION * feat.get("location_score", 0.5)
        final = total * feat.get("behavioral_multiplier", 0.75)
        final_results.append((c, feat, final, semantic))

    final_results.sort(key=lambda x: x[2], reverse=True)
    top_results = final_results[:min(top_n, 100)]

    col2.metric("Ranked", len(top_results))
    col3.metric("Time", f"{time.time()-t0:.1f}s")

    from reasoning import generate_reasoning
    rows = []
    for rank_pos, (c, feat, score, sem) in enumerate(top_results, 1):
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank_pos,
            "score": round(score, 6),
            "reasoning": generate_reasoning(c, feat, sem, score, rank_pos),
            "title": c.get("profile", {}).get("current_title", ""),
            "company": c.get("profile", {}).get("current_company", ""),
            "yoe": c.get("profile", {}).get("years_of_experience", 0),
            "location": c.get("profile", {}).get("location", ""),
        })

    import pandas as pd
    st.subheader(f"Top {len(rows)} Candidates")
    st.dataframe(pd.DataFrame([{
        "Rank": r["rank"], "Score": f"{r['score']:.4f}",
        "Title": r["title"], "Company": r["company"],
        "YoE": r["yoe"], "Location": r["location"],
        "Reasoning": r["reasoning"][:100] + "...",
    } for r in rows]), use_container_width=True, height=400)

    if excluded:
        with st.expander(f"Excluded ({len(excluded)})"):
            for cid, reason in list(excluded.items())[:20]:
                st.text(f"{cid}: {reason}")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])

    st.download_button("📥 Download submission.csv", data=buf.getvalue().encode("utf-8"), file_name="submission.csv", mime="text/csv")
    st.success(f"✅ Done in {time.time()-t0:.1f}s")
