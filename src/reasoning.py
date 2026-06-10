from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_TODAY = datetime.now()

PRODUCTION_SIGNALS = [
    "retrieval", "ranking", "recommendation", "search", "embedding", "vector",
    "pinecone", "faiss", "elasticsearch", "opensearch", "weaviate", "milvus",
    "qdrant", "hybrid search", "semantic search", "rag", "rerank",
    "a/b test", "ndcg", "mrr", "evaluation", "xgboost", "lightgbm",
    "learning to rank", "feature engineering", "model serving", "production",
]

RETRIEVAL_SKILLS = {
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "elasticsearch",
    "opensearch", "embeddings", "sentence-transformers", "sentence transformers",
    "bge", "retrieval", "information retrieval", "semantic search", "vector",
    "rag", "reranking", "ranking", "hybrid search", "bm25", "xgboost",
    "lightgbm", "learning to rank", "recommendation", "recommendation systems",
}


def _days_since(date_str: str) -> int:
    try:
        return max(0, (_TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days)
    except Exception:
        return 9999


def _find_key_skills(candidate: dict) -> list[tuple[str, int]]:
    return [
        (s["name"], s.get("duration_months", 0))
        for s in candidate.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
        and any(kw in s.get("name", "").lower() for kw in RETRIEVAL_SKILLS)
    ][:5]


def _find_production_evidence(candidate: dict) -> list[str]:
    evidence = []
    for job in candidate.get("career_history", [])[:4]:
        desc = job.get("description", "").lower()
        hits = [sig for sig in PRODUCTION_SIGNALS if sig in desc]
        if hits:
            evidence.append(f"{job.get('title')} at {job.get('company')}: {', '.join(hits[:3])}")
    return evidence


def _build_candidate_summary_for_llm(candidate: dict, features: dict) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    key_skills = _find_key_skills(candidate)
    production_evidence = _find_production_evidence(candidate)

    career_lines = []
    for job in career[:4]:
        desc = job.get("description", "").strip()[:300]
        career_lines.append(
            f"- {job['title']} at {job['company']} ({job.get('duration_months',0)}mo, "
            f"{job.get('industry','unknown industry')}): {desc}"
        )

    adv_skills = [
        f"{s['name']} ({s.get('proficiency')}, {s.get('duration_months',0)}mo, "
        f"{s.get('endorsements',0)} endorsements)"
        for s in skills if s.get("proficiency") in ("advanced", "expert")
    ][:10]

    assess = signals.get("skill_assessment_scores", {})

    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_str = f"{sal.get('min',0)}-{sal.get('max',0)} LPA" if isinstance(sal, dict) else str(sal)

    return f"""
CANDIDATE PROFILE:
Title: {profile.get('current_title')} at {profile.get('current_company')} ({profile.get('current_company_size')})
Experience: {profile.get('years_of_experience')} years | Location: {profile.get('location')}, {profile.get('country')}
Industry: {profile.get('current_industry')}
Summary: {profile.get('summary','')[:400]}

CAREER HISTORY:
{chr(10).join(career_lines)}

SKILLS (advanced/expert only):
{', '.join(adv_skills)}

ASSESSMENT SCORES: {json.dumps(assess)}

SIGNALS:
- Last active: {signals.get('last_active_date')} ({_days_since(signals.get('last_active_date',''))}d ago)
- Open to work: {signals.get('open_to_work_flag')}
- Notice period: {signals.get('notice_period_days')}d
- Recruiter response rate: {signals.get('recruiter_response_rate')}
- Saved by recruiters (30d): {signals.get('saved_by_recruiters_30d')}
- GitHub score: {signals.get('github_activity_score')}
- Expected salary: {sal_str}
- Willing to relocate: {signals.get('willing_to_relocate')}
- Profile completeness: {signals.get('profile_completeness_score')}%

COMPUTED SCORES:
- Career arc score: {features.get('career_arc_score',0):.3f}
- Production score: {features.get('production_score',0):.3f}
- Product company ratio: {features.get('product_ratio',0):.3f}
- Availability score: {features.get('availability_score',0):.3f}
- Must-have skill count: {features.get('must_have_count',0)}

KEY PRODUCTION EVIDENCE FOUND:
{chr(10).join(production_evidence) if production_evidence else 'None found'}
""".strip()


def generate_llm_reasoning_batch(
    candidates_data: list[tuple[str, dict, dict, float, int]],
    cache_path: Path,
) -> dict[str, str]:
    """
    Generate LLM reasoning for a batch of candidates.
    candidates_data: list of (candidate_id, candidate, features, final_score, rank)
    Returns dict of candidate_id -> reasoning string.
    Saves to cache so ranking step never calls the API.
    """
    existing_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing_cache = json.load(f)

    results = dict(existing_cache)
    to_generate = [(cid, c, feat, score, rank) for cid, c, feat, score, rank in candidates_data
                   if cid not in existing_cache]

    if not to_generate:
        print(f"[reasoning] All {len(candidates_data)} reasonings found in cache")
        return results

    print(f"[reasoning] Generating LLM reasoning for {len(to_generate)} candidates ...")

    try:
        import urllib.request
        import urllib.error
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("[reasoning] ERROR: ANTHROPIC_API_KEY environment variable not set.")
            print("[reasoning] Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
            print("[reasoning] Falling back to template reasoning for all candidates.")
            for cid, candidate, features, final_score, rank in to_generate:
                results[cid] = _template_reasoning(candidate, features, final_score, rank)
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)
            return results

        print(f"[reasoning] API key found: {api_key[:12]}...")

        for i, (cid, candidate, features, final_score, rank) in enumerate(to_generate, 1):
            candidate_summary = _build_candidate_summary_for_llm(candidate, features)

            prompt = f"""You are a senior technical recruiter evaluating candidates for this role:

JOB: Senior AI Engineer — Founding Team at a Series A startup
ROLE SUMMARY: Build and own the AI ranking/retrieval infrastructure from scratch.
REQUIREMENTS: 5-9yr experience, production embeddings + vector DB systems, learning-to-rank,
evaluation frameworks (NDCG/MRR), product company background (not pure services).
LOCATION: Pune or Noida (or willing to relocate). No visa sponsorship.

{candidate_summary}

Write a 2-3 sentence recruiter assessment for this candidate at rank #{rank} out of 100.

Rules:
- Reference SPECIFIC facts from their profile (company names, skill durations, actual experience)
- Be honest — acknowledge gaps for lower-ranked candidates
- Sound like a real senior recruiter wrote this, not a template
- For rank 1-10: strong, specific endorsement
- For rank 11-30: positive with one specific concern
- For rank 31-60: balanced, honest about gaps
- For rank 61-100: honest that this is marginal, specific gaps stated
- Maximum 3 sentences. No bullet points. No fluff.

Write only the assessment, nothing else."""

            payload = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": api_key,
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    reasoning = data["content"][0]["text"].strip()
                    results[cid] = reasoning
                    print(f"  [{i}/{len(to_generate)}] {cid}: {reasoning[:80]}...")
            except Exception as e:
                print(f"  [{i}/{len(to_generate)}] {cid}: API error ({e}), using fallback")
                results[cid] = _template_reasoning(candidate, features, final_score, rank)

            # Save after every candidate so progress is not lost on interruption
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    except Exception as e:
        print(f"[reasoning] LLM generation failed: {e}. Using template fallback for all.")
        for cid, candidate, features, final_score, rank in to_generate:
            if cid not in results:
                results[cid] = _template_reasoning(candidate, features, final_score, rank)

    print(f"[reasoning] Done. Cache saved to {cache_path}")
    return results


def _template_reasoning(candidate: dict, features: dict, final_score: float, rank: int) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = round(profile.get("years_of_experience", 0), 1)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    loc_str = f"{location}, {country}".strip(", ")
    notice = signals.get("notice_period_days", 90)
    days_ago = _days_since(signals.get("last_active_date", ""))
    open_to_work = signals.get("open_to_work_flag", False)
    rr = signals.get("recruiter_response_rate", 0)

    key_skills = _find_key_skills(candidate)
    prod_evidence = _find_production_evidence(candidate)
    skill_str = f"; {key_skills[0][0]} ({key_skills[0][1]}mo)" if key_skills else ""
    prod_str = f"; {prod_evidence[0]}" if prod_evidence else ""

    if rank <= 10:
        avail = f"actively seeking ({notice}d notice)" if open_to_work and notice <= 60 else f"{notice}d notice"
        return (
            f"{yoe}yr {title} at {company} ({loc_str}){skill_str}{prod_str}. "
            f"Strong match on JD's product-company + production retrieval requirements. "
            f"{avail}, last active {days_ago}d ago."
        )
    elif rank <= 30:
        concern = (f"Notice period {notice}d." if notice > 90
                   else f"Inactive {days_ago}d." if days_ago > 60
                   else f"Response rate {rr:.0%}." if rr < 0.3
                   else "")
        return (
            f"{yoe}yr {title} at {company} ({loc_str}){skill_str}{prod_str}. "
            f"Good retrieval/ranking signals. {concern}"
        )
    elif rank <= 60:
        gaps = []
        if features.get("product_ratio", 1) < 0.4:
            gaps.append("services-heavy background")
        if features.get("production_score", 0) < 0.3:
            gaps.append("limited production ML evidence")
        if days_ago > 90:
            gaps.append(f"inactive {days_ago}d")
        gap_str = "; ".join(gaps) if gaps else "partial skill match"
        return (
            f"{yoe}yr {title} at {company} ({loc_str}). "
            f"Partial JD match. Concerns: {gap_str}."
        )
    else:
        gaps = []
        if not key_skills:
            gaps.append("no core retrieval skills at advanced level")
        if features.get("production_score", 0) < 0.2:
            gaps.append("no production ML evidence in career history")
        if features.get("product_ratio", 0) < 0.3:
            gaps.append("entirely services background")
        gap_str = "; ".join(gaps) if gaps else "adjacent profile"
        return (
            f"{yoe}yr {title} at {company} ({loc_str}). "
            f"Marginal fit — completes top-100 shortlist. Gaps: {gap_str}."
        )


def generate_reasoning(
    candidate: dict,
    features: dict,
    semantic_score: float,
    final_score: float,
    rank: int,
    cache: dict | None = None,
) -> str:
    cid = candidate.get("candidate_id", "")
    if cache and cid in cache:
        return cache[cid]
    return _template_reasoning(candidate, features, final_score, rank)
