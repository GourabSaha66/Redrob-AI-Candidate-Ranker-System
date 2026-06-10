from __future__ import annotations
from typing import Any


def _deduplicate_descriptions(career: list[dict]) -> list[dict]:
    seen, deduped = set(), []
    for job in career:
        desc = job.get("description", "").strip()
        if desc and desc not in seen:
            seen.add(desc)
            deduped.append(job)
        elif not desc:
            deduped.append(job)
    return deduped


def build_candidate_text(candidate: dict[str, Any], mode: str = "full") -> str:
    profile = candidate.get("profile", {})
    career = _deduplicate_descriptions(candidate.get("career_history", []))
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    parts = []

    if mode in ("full", "summary"):
        if profile.get("headline"):
            parts.append(f"Role: {profile['headline'].strip()}")
        if profile.get("summary"):
            parts.append(f"Summary: {profile['summary'].strip()[:200]}")

    if mode in ("full", "career"):
        for job in career[:5]:
            desc = job.get("description", "").strip()
            if desc:
                years = round(job.get("duration_months", 0) / 12, 1)
                parts.append(f"Work ({years}yr as {job.get('title','')} at {job.get('company','')}): {desc[:300]}")

    if mode == "full":
        advanced = [s["name"] for s in skills if s.get("proficiency") == "advanced"]
        intermediate = [s["name"] for s in skills if s.get("proficiency") == "intermediate"]
        if advanced:
            parts.append(f"Advanced: {', '.join(advanced[:10])}")
        if intermediate:
            parts.append(f"Intermediate: {', '.join(intermediate[:8])}")

        saved = signals.get("saved_by_recruiters_30d", 0)
        gh = signals.get("github_activity_score", -1)
        if saved > 0:
            parts.append(f"Saved by {saved} recruiters recently")
        if gh > 30:
            parts.append(f"Active GitHub contributor score {gh}")

    return " | ".join(parts)[:2000]
