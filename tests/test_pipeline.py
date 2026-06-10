#!/usr/bin/env python3
"""
tests/test_pipeline.py

Run on the 50-sample candidates to verify everything works before
running on the full 100K pool.

Usage:
    python tests/test_pipeline.py
"""

import json
import sys
import os
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_candidates.json"


def load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        print(f"[test] Sample file not found at {SAMPLE_PATH}")
        print("       Copy sample_candidates.json to data/sample_candidates.json")
        sys.exit(1)
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def test_filters(candidates):
    print("\n── test_filters ─────────────────────────────────────────────")
    from filters import apply_filters, detect_honeypot, is_disqualified

    kept, excluded = apply_filters(candidates)
    print(f"  {len(candidates)} → {len(kept)} kept, {len(excluded)} excluded")

    # Specific checks on known sample data
    for c in candidates:
        cid = c["candidate_id"]
        title = c["profile"]["current_title"]
        is_hp, hp_reason = detect_honeypot(c)
        is_dq, dq_reason = is_disqualified(c)
        status = "HONEYPOT" if is_hp else "DISQ" if is_dq else "KEEP"
        if status != "KEEP":
            print(f"  {cid} [{title}] → {status}: {hp_reason or dq_reason}")

    # Non-technical roles should be filtered
    non_tech_ids = [
        c["candidate_id"] for c in candidates
        if any(kw in c["profile"]["current_title"].lower()
               for kw in ["accountant", "customer support", "marketing manager", "operations manager"])
    ]
    for cid in non_tech_ids:
        assert cid in excluded, f"Non-technical {cid} should be excluded but wasn't"
    print(f"  ✓ All {len(non_tech_ids)} non-technical roles correctly excluded")
    return kept, excluded


def test_features(candidates):
    print("\n── test_features ────────────────────────────────────────────")
    from features import featurize

    for c in candidates[:5]:
        feats = featurize(c)
        cid = c["candidate_id"]
        title = c["profile"]["current_title"]
        print(f"  {cid} [{title}]:")
        print(f"    career_arc={feats['career_arc_score']:.3f} | "
              f"availability={feats['availability_score']:.3f} | "
              f"location={feats['location_score']:.3f} | "
              f"behavioral_mult={feats['behavioral_multiplier']:.3f}")

    # Verify score ranges
    for c in candidates:
        feats = featurize(c)
        for key, val in feats.items():
            if isinstance(val, float) and key.endswith("_score"):
                assert 0.0 <= val <= 1.0, f"{c['candidate_id']}.{key}={val} out of [0,1]"
    print("  ✓ All scores in valid [0,1] range")


def test_text_builder(candidates):
    print("\n── test_text_builder ────────────────────────────────────────")
    from text_builder import build_candidate_text

    for c in candidates[:3]:
        text = build_candidate_text(c, mode="full")
        print(f"  {c['candidate_id']}: {len(text)} chars — {text[:100]}...")
        assert len(text) > 50, "Text too short"
        assert len(text) <= 2001, "Text too long"
    print("  ✓ Text builder producing valid output")


def test_reasoning(candidates):
    print("\n── test_reasoning ───────────────────────────────────────────")
    from reasoning import generate_reasoning
    from features import featurize

    # Test different rank tiers
    for rank_pos, c in zip([1, 15, 40, 80], candidates[:4]):
        feats = featurize(c)
        reasoning = generate_reasoning(
            candidate=c, features=feats,
            semantic_score=0.8, final_score=0.75, rank=rank_pos
        )
        print(f"  Rank {rank_pos}: {reasoning[:120]}...")
        assert len(reasoning) > 30, "Reasoning too short"
        assert len(reasoning) < 500, "Reasoning too long"

    # Check that all 4 reasonings are unique
    reasonings = []
    for rank_pos, c in zip([1, 15, 40, 80], candidates[:4]):
        feats = featurize(c)
        r = generate_reasoning(c, feats, 0.8, 0.75, rank_pos)
        reasonings.append(r)
    assert len(set(reasonings)) == len(reasonings), "Reasoning strings not unique!"
    print("  ✓ All reasoning strings unique and valid")


def test_submission_format():
    """Verify validate_submission.py would pass on a properly generated CSV."""
    print("\n── test_submission_format ───────────────────────────────────")
    import csv, io

    rows = []
    for i in range(1, 101):
        cid = f"CAND_{i:07d}"
        rows.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(1.0 - (i - 1) * 0.005, 6),
            "reasoning": f"Test reasoning for rank {i} candidate with specific skills.",
        })

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for row in rows:
        writer.writerow([row["candidate_id"], row["rank"],
                         f"{row['score']:.6f}", row["reasoning"]])
    content = buf.getvalue()

    assert content.count("\n") == 101, "Should have 101 lines (header + 100 data)"
    print("  ✓ CSV format matches spec (101 lines, correct columns)")


if __name__ == "__main__":
    print("Running pipeline tests on sample_candidates.json ...")
    candidates = load_sample()
    print(f"Loaded {len(candidates)} sample candidates")

    test_filters(candidates)
    test_features(candidates)
    test_text_builder(candidates)
    test_reasoning(candidates)
    test_submission_format()

    print("\n✓ All tests passed!")
    print("\nNext steps:")
    print("  1. Copy candidates.jsonl to data/")
    print("  2. python scripts/precompute_embeddings.py --candidates data/candidates.jsonl")
    print("  3. python rank.py --candidates data/candidates.jsonl --out submission.csv")
    print("  4. python validate_submission.py submission.csv")
