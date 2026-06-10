from __future__ import annotations
from datetime import datetime
from typing import Any
from config import PURE_SERVICES_COMPANIES, MUST_HAVE_SKILLS, TIER1_INDIA_LOCATIONS, PRODUCT_INDUSTRIES

_TODAY = datetime.now()


def _norm(s: str) -> str:
    return s.lower().strip()


def _days_since(date_str: str) -> int:
    try:
        return max(0, (_TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days)
    except Exception:
        return 9999


def _parse_salary(salary) -> tuple[float, float]:
    if isinstance(salary, dict):
        return float(salary.get("min", 0) or 0), float(salary.get("max", 0) or 0)
    import re
    try:
        nums = re.findall(r'\d+\.?\d*', str(salary))
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        elif len(nums) == 1:
            v = float(nums[0])
            return v, v
    except Exception:
        pass
    return 0.0, 0.0


def career_features(candidate: dict[str, Any]) -> dict[str, float]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    yoe = float(profile.get("years_of_experience", 0))
    title = _norm(profile.get("current_title", ""))

    # Product company ratio — weighted by months
    total_months = sum(j.get("duration_months", 0) for j in career) or 1
    services_months = sum(
        j.get("duration_months", 0) for j in career
        if any(svc in _norm(j.get("company", "")) for svc in PURE_SERVICES_COMPANIES)
    )
    product_ratio = max(0.0, 1.0 - services_months / total_months)

    # Product industry bonus — working at Swiggy/Zomato/Uber/Flipkart is a strong signal
    product_industry_months = sum(
        j.get("duration_months", 0) for j in career
        if any(ind in _norm(j.get("industry", "")) for ind in PRODUCT_INDUSTRIES)
    )
    product_industry_score = min(1.0, product_industry_months / 36.0)

    # IC vs management
    mgmt = ["director", "vp ", "vice president", "cto", "ceo", "head of", "chief ", "president"]
    eng = ["engineer", "scientist", "researcher", "developer", "architect", "analyst", "lead", "staff", "principal"]
    is_ic = any(e in title for e in eng) and not any(m in title for m in mgmt)
    is_mgmt_only = any(m in title for m in mgmt) and not any(e in title for e in eng)

    # Tenure stability
    past_jobs = [j for j in career if not j.get("is_current")]
    avg_tenure = sum(j.get("duration_months", 0) for j in past_jobs) / len(past_jobs) if past_jobs else 0
    tenure_score = min(1.0, avg_tenure / 24.0) if past_jobs else 0.7

    # Production ML depth — keyword hits in career descriptions
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    prod_keywords = [
        "deployed", "production", "real users", "a/b test", "latency", "serving",
        "inference", "pipeline", "monitoring", "drift", "retrieval", "embedding",
        "ranking", "search", "recommendation", "xgboost", "lightgbm",
        "ndcg", "mrr", "learning to rank", "evaluation framework",
        "vector", "faiss", "pinecone", "semantic", "hybrid search",
    ]
    production_score = min(1.0, sum(1 for kw in prod_keywords if kw in all_desc) / 10.0)

    # Pre-LLM ML experience (shows real depth, not LangChain tourist)
    pre_llm_months = 0
    for job in career:
        try:
            end = job.get("end_date") or "2030-01-01"
            if datetime.strptime(end, "%Y-%m-%d").year <= 2022:
                desc = job.get("description", "").lower()
                if any(kw in desc for kw in ["ml", "machine learning", "model", "retrieval", "ranking", "nlp", "search", "recommendation"]):
                    pre_llm_months += job.get("duration_months", 0)
        except Exception:
            pass
    pre_llm_score = min(1.0, pre_llm_months / 24.0)

    # YoE fit — 5-9yr is the sweet spot per JD
    if 5.0 <= yoe <= 9.0:
        yoe_fit = 1.0
    elif 4.0 <= yoe < 5.0:
        yoe_fit = 0.85
    elif 9.0 < yoe <= 12.0:
        yoe_fit = 0.90
    elif yoe < 4.0:
        yoe_fit = max(0.0, yoe / 4.0)
    else:
        yoe_fit = 0.75

    # Skill scoring — use duration_months and endorsements on skills, not just name
    # This is the key differentiator: a skill with 60mo duration + 40 endorsements
    # is worth far more than a skill listed with 0 context
    assessment_scores = signals.get("skill_assessment_scores", {})
    must_have_score = 0.0
    must_have_count = 0
    total_relevant_months = 0

    for s in skills:
        name = s.get("name", "")
        name_lower = _norm(name)
        proficiency = _norm(s.get("proficiency", ""))
        duration = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        is_must = any(kw in name_lower for kw in MUST_HAVE_SKILLS)
        if not is_must:
            continue

        must_have_count += 1
        total_relevant_months += duration

        # Base score from proficiency
        base = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(proficiency, 0.4)

        # Duration modifier — 24+ months of using this skill = strong signal
        duration_mod = min(1.3, 1.0 + duration / 120.0)

        # Endorsement modifier — social proof
        endorse_mod = min(1.2, 1.0 + endorsements / 100.0)

        # Assessment credibility — if they took a test, use it
        assess = assessment_scores.get(name, None)
        if assess is not None:
            assess_mod = assess / 100.0
            # Penalize if they claim advanced but scored < 40
            if proficiency == "advanced" and assess < 40:
                assess_mod = 0.5
            elif proficiency == "expert" and assess < 50:
                assess_mod = 0.6
        else:
            assess_mod = 0.7  # no test = slight uncertainty

        skill_val = base * duration_mod * endorse_mod * assess_mod
        must_have_score += skill_val

    # Normalize must_have_score
    normalized_skill_score = min(1.0, must_have_score / 6.0)
    relevant_skill_months_score = min(1.0, total_relevant_months / 60.0)

    # GitHub
    gh = signals.get("github_activity_score", -1)
    gh_score = 0.25 if gh == -1 else min(1.0, gh / 60.0)

    # Market demand — saved by recruiters and search appearances
    saved_score = min(1.0, signals.get("saved_by_recruiters_30d", 0) / 10.0)
    search_score = min(1.0, signals.get("search_appearance_30d", 0) / 500.0)

    # Salary fit
    sal_min, sal_max = _parse_salary(signals.get("expected_salary_range_inr_lpa", ""))
    if sal_min == 0 and sal_max == 0:
        salary_fit = 0.7
    elif sal_min > 120:
        salary_fit = 0.3
    elif sal_max < 15:
        salary_fit = 0.5
    else:
        salary_fit = 1.0

    career_arc_score = (
        0.20 * product_ratio
        + 0.08 * product_industry_score
        + 0.10 * (1.0 if is_ic else 0.3 if not is_mgmt_only else 0.0)
        + 0.15 * tenure_score
        + 0.22 * production_score
        + 0.10 * pre_llm_score
        + 0.07 * yoe_fit
        + 0.08 * normalized_skill_score
    )

    return {
        "career_arc_score":           round(career_arc_score, 4),
        "product_ratio":              round(product_ratio, 4),
        "product_industry_score":     round(product_industry_score, 4),
        "is_ic":                      float(is_ic),
        "tenure_score":               round(tenure_score, 4),
        "production_score":           round(production_score, 4),
        "pre_llm_score":              round(pre_llm_score, 4),
        "yoe_fit":                    round(yoe_fit, 4),
        "normalized_skill_score":     round(normalized_skill_score, 4),
        "relevant_skill_months_score":round(relevant_skill_months_score, 4),
        "must_have_count":            must_have_count,
        "gh_score":                   round(gh_score, 4),
        "saved_score":                round(saved_score, 4),
        "search_score":               round(search_score, 4),
        "salary_fit":                 round(salary_fit, 4),
    }


def behavioral_features(candidate: dict[str, Any]) -> dict[str, float]:
    s = candidate.get("redrob_signals", {})

    days_inactive = _days_since(s.get("last_active_date", "2020-01-01"))
    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.90
    elif days_inactive <= 60:
        recency = 0.80
    elif days_inactive <= 90:
        recency = 0.70
    elif days_inactive <= 180:
        recency = 0.50
    else:
        recency = max(0.0, 1.0 - days_inactive / 400.0)

    open_to_work = 1.0 if s.get("open_to_work_flag") else 0.45
    apps_30d = min(1.0, s.get("applications_submitted_30d", 0) / 5.0)
    seeking_score = 0.65 * open_to_work + 0.35 * apps_30d

    rr = s.get("recruiter_response_rate", 0.0)
    speed = max(0.0, 1.0 - s.get("avg_response_time_hours", 999) / 72.0)
    responsiveness = 0.70 * rr + 0.30 * speed

    notice = s.get("notice_period_days", 90)
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.95
    elif notice <= 60:
        notice_score = 0.80
    elif notice <= 90:
        notice_score = 0.60
    else:
        notice_score = max(0.25, 1.0 - notice / 200.0)

    completeness = s.get("profile_completeness_score", 50) / 100.0
    interview_cr = s.get("interview_completion_rate", 0.5)
    offer_ar = s.get("offer_acceptance_rate", -1)
    offer_ar = 0.5 if offer_ar == -1 else offer_ar
    verified = float(s.get("verified_email", False)) * 0.5 + float(s.get("verified_phone", False)) * 0.5
    linkedin = float(s.get("linkedin_connected", False)) * 0.3
    connections = min(1.0, s.get("connection_count", 0) / 300.0)
    endorsements_total = min(1.0, s.get("endorsements_received", 0) / 100.0)

    engagement = (
        0.20 * completeness
        + 0.25 * interview_cr
        + 0.15 * offer_ar
        + 0.10 * verified
        + 0.10 * linkedin
        + 0.10 * connections
        + 0.10 * endorsements_total
    )

    availability = (
        0.35 * recency
        + 0.25 * seeking_score
        + 0.25 * responsiveness
        + 0.15 * notice_score
    )

    # Multiplicative behavioral multiplier [0.45, 1.0]
    # Dead candidates (inactive + unresponsive + long notice) get heavily penalized
    raw = 0.45 * availability + 0.35 * responsiveness + 0.20 * notice_score
    behavioral_multiplier = 0.45 + 0.55 * raw

    return {
        "availability_score":    round(availability, 4),
        "responsiveness_score":  round(responsiveness, 4),
        "notice_score":          round(notice_score, 4),
        "engagement_score":      round(engagement, 4),
        "recency_score":         round(recency, 4),
        "behavioral_multiplier": round(behavioral_multiplier, 4),
        "days_inactive":         days_inactive,
        "notice_period_days":    notice,
    }


def location_features(candidate: dict[str, Any]) -> dict[str, float]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = _norm(profile.get("location", ""))
    country = _norm(profile.get("country", ""))
    will_relocate = signals.get("willing_to_relocate", False)
    preferred_mode = _norm(signals.get("preferred_work_mode", ""))

    mode_ok = "hybrid" in preferred_mode or "onsite" in preferred_mode or "flexible" in preferred_mode
    mode_score = 1.0 if mode_ok else 0.85

    is_tier1 = any(city in location for city in TIER1_INDIA_LOCATIONS)
    is_india = country in ("india", "in") or "india" in location

    if is_tier1:
        location_score = 1.0 * mode_score
    elif is_india:
        location_score = (0.82 if will_relocate else 0.62) * mode_score
    elif will_relocate:
        location_score = 0.65 * mode_score
    else:
        location_score = 0.30

    return {
        "location_score":  round(location_score, 4),
        "is_tier1_india":  float(is_tier1),
        "is_india":        float(is_india),
        "will_relocate":   float(will_relocate),
    }


def featurize(candidate: dict[str, Any]) -> dict[str, float]:
    feats = {}
    feats.update(career_features(candidate))
    feats.update(behavioral_features(candidate))
    feats.update(location_features(candidate))
    return feats
