from __future__ import annotations
import re
from datetime import datetime
from typing import Any
from config import PURE_SERVICES_COMPANIES, RED_FLAG_SKILLS, MIN_YEARS_EXPERIENCE, HONEYPOT_ASSESSMENT_THRESHOLD


def _norm(s: str) -> str:
    return s.lower().strip()


def _is_services_company(company: str) -> bool:
    cn = _norm(company)
    return any(svc in cn for svc in PURE_SERVICES_COMPANIES)


HARD_NON_TECHNICAL = {
    "operations manager", "customer support", "marketing manager", "accountant",
    "hr manager", "human resources", "content writer", "graphic designer",
    "project manager", "business analyst", "sales", "finance", "legal",
    "recruiter", "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer", "manufacturing engineer", "production engineer",
    "industrial engineer", "structural engineer", "supply chain", "logistics",
    "procurement", "administrative",
}

CONDITIONAL_TITLES = {
    "frontend engineer", "frontend developer", "java developer", "java engineer",
    ".net developer", ".net engineer", "ios developer", "android developer",
    "mobile developer", "ui developer", "react developer", "angular developer",
    "web developer", "full stack developer", "full stack engineer",
    "devops engineer", "sre engineer", "site reliability", "network engineer",
    "database administrator", "dba",
}

ML_CAREER_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "nlp", "natural language",
    "recommendation", "ranking", "retrieval", "embedding", "vector", "search",
    "model training", "model deployment", "feature engineering", "xgboost",
    "pytorch", "tensorflow", "transformers", "llm", "rag", "fine-tuning",
    "a/b test", "inference", "model serving", "ml pipeline",
]

ML_SKILL_KEYWORDS = [
    "pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm",
    "transformers", "bert", "gpt", "llm", "rag", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "embeddings",
    "sentence-transformers", "nlp", "machine learning", "deep learning",
    "recommendation", "ranking", "retrieval",
]


def _has_ml_evidence(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    career_hits = sum(1 for kw in ML_CAREER_KEYWORDS if kw in all_desc)
    skill_names = {_norm(s.get("name", "")) for s in skills}
    skill_hits = sum(1 for kw in ML_SKILL_KEYWORDS if any(kw in sn for sn in skill_names))
    return career_hits >= 3 or skill_hits >= 4


def _has_identical_descriptions(career: list) -> bool:
    descs = [j.get("description", "").strip() for j in career if j.get("description")]
    return len(descs) >= 3 and len(set(descs)) == 1


def detect_honeypot(candidate: dict[str, Any]) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    claimed_yoe = profile.get("years_of_experience", 0)
    if claimed_yoe > 30:
        return True, f"Impossible YoE={claimed_yoe}"

    advanced_skills = [s for s in skills if _norm(s.get("proficiency", "")) == "advanced"]
    assessment_scores = signals.get("skill_assessment_scores", {})
    if len(advanced_skills) >= 6 and assessment_scores:
        avg = sum(assessment_scores.values()) / len(assessment_scores)
        if avg < HONEYPOT_ASSESSMENT_THRESHOLD:
            return True, f"Claims advanced in {len(advanced_skills)} skills but avg assessment={avg:.1f}"

    non_current = [j for j in career if not j.get("is_current")]
    current_months = next((j.get("duration_months", 0) for j in career if j.get("is_current")), 0)
    total_actual = sum(j.get("duration_months", 0) for j in non_current) + current_months
    claimed_months = claimed_yoe * 12
    if claimed_months > 0 and total_actual < claimed_months - 60:
        return True, f"Career history sums to {total_actual/12:.1f}yr but claims {claimed_yoe}yr"

    if _has_identical_descriptions(career):
        return True, "All career descriptions are identical"

    return False, ""


def is_disqualified(candidate: dict[str, Any]) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = _norm(profile.get("current_title", ""))
    yoe = profile.get("years_of_experience", 0)

    if yoe < MIN_YEARS_EXPERIENCE:
        return True, f"Too junior: {yoe}yr"

    for nt in HARD_NON_TECHNICAL:
        if nt in title:
            return True, f"Non-technical role: {profile.get('current_title')}"

    is_conditional = any(ct in title for ct in CONDITIONAL_TITLES)
    if is_conditional and not _has_ml_evidence(candidate):
        return True, f"'{profile.get('current_title')}' with no ML evidence in career or skills"

    has_product = any(not _is_services_company(j.get("company", "")) for j in career)
    if not has_product and len(career) >= 2:
        return True, "100% services career"

    skill_names = {_norm(s.get("name", "")) for s in skills}
    red_count = sum(1 for rf in RED_FLAG_SKILLS if rf in skill_names)
    has_nlp_ir = any(kw in skill_names for kw in {"nlp", "retrieval", "embeddings", "rag", "ranking"})
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    has_nlp_career = any(kw in all_desc for kw in ["nlp", "retrieval", "ranking", "embedding", "search"])
    if red_count >= 4 and not has_nlp_ir and not has_nlp_career:
        return True, f"CV/Speech specialist with no IR/NLP signal"

    return False, ""


def apply_filters(candidates: list[dict]) -> tuple[list[dict], dict[str, str]]:
    kept, excluded = [], {}
    for c in candidates:
        cid = c["candidate_id"]
        is_hp, reason = detect_honeypot(c)
        if is_hp:
            excluded[cid] = f"HONEYPOT: {reason}"
            continue
        is_dq, reason = is_disqualified(c)
        if is_dq:
            excluded[cid] = f"DISQUALIFIED: {reason}"
            continue
        kept.append(c)
    honeypots = sum(1 for v in excluded.values() if v.startswith("HONEYPOT"))
    print(f"[filter] {len(candidates):,} in → {len(kept):,} kept, {len(excluded):,} excluded ({honeypots} honeypots)")
    return kept, excluded
