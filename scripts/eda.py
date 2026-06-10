import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.artifacts)

    with open(out_dir / "candidate_ids.json") as f:
        ids = json.load(f)
    with open(out_dir / "candidate_features.json") as f:
        features = json.load(f)
    with open(out_dir / "candidate_slim.json") as f:
        slim = json.load(f)
    with open(out_dir / "excluded_candidates.json") as f:
        excluded = json.load(f)

    print(f"Total candidates: {len(ids) + len(excluded):,}")
    print(f"Kept after filter: {len(ids):,}")
    print(f"Excluded: {len(excluded):,}")
    honeypots = sum(1 for v in excluded.values() if "HONEYPOT" in v)
    disq = sum(1 for v in excluded.values() if "DISQUALIFIED" in v)
    print(f"  Honeypots: {honeypots:,}")
    print(f"  Disqualified: {disq:,}")

    excl_reasons = Counter()
    for v in excluded.values():
        reason = v.split(": ", 1)[-1]
        if "Non-technical" in reason:
            excl_reasons["Non-technical role"] += 1
        elif "Too junior" in reason:
            excl_reasons["Too junior"] += 1
        elif "100% services" in reason:
            excl_reasons["100% services career"] += 1
        elif "no ML evidence" in reason:
            excl_reasons["No ML evidence"] += 1
        elif "HONEYPOT" in v:
            excl_reasons["Honeypot"] += 1
        else:
            excl_reasons["Other"] += 1
    print("\nExclusion breakdown:")
    for reason, count in excl_reasons.most_common():
        print(f"  {reason}: {count:,}")

    feat_vals = {k: [features[i][k] for i in ids if k in features[i]] for k in [
        "career_arc_score", "production_score", "availability_score",
        "location_score", "must_have_count", "behavioral_multiplier",
        "product_ratio", "pre_llm_score", "tenure_score", "assessment_credibility",
        "salary_fit", "saved_score", "gh_score"
    ]}

    print("\n=== Feature distributions ===")
    print(f"{'Feature':<25} {'Mean':>6} {'Median':>7} {'Min':>6} {'Max':>6} {'Stdev':>7}")
    print("-" * 60)
    import statistics
    for name, vals in feat_vals.items():
        if not vals:
            continue
        print(f"{name:<25} {statistics.mean(vals):>6.3f} {statistics.median(vals):>7.3f} "
              f"{min(vals):>6.3f} {max(vals):>6.3f} {statistics.stdev(vals):>7.3f}")

    print("\n=== Location breakdown ===")
    loc_counts = Counter(
        "Tier-1 India" if features[i]["is_tier1_india"] else
        "India (other)" if features[i]["is_india"] else
        "Outside India (relocate)" if features[i]["will_relocate"] else
        "Outside India (no relocate)"
        for i in ids
    )
    for loc, count in loc_counts.most_common():
        pct = count / len(ids) * 100
        print(f"  {loc}: {count:,} ({pct:.1f}%)")

    print("\n=== Must-have skill count distribution ===")
    must_counts = Counter(features[i]["must_have_count"] for i in ids)
    for count in sorted(must_counts.keys()):
        bar = "█" * min(40, must_counts[count] // 10)
        print(f"  {count} skills: {must_counts[count]:>5,}  {bar}")

    print("\n=== Production score distribution ===")
    buckets = Counter()
    for i in ids:
        p = features[i]["production_score"]
        if p == 0:
            buckets["0.0 (no signals)"] += 1
        elif p < 0.25:
            buckets["0.01-0.24 (weak)"] += 1
        elif p < 0.5:
            buckets["0.25-0.49 (moderate)"] += 1
        elif p < 0.75:
            buckets["0.50-0.74 (strong)"] += 1
        else:
            buckets["0.75-1.0 (excellent)"] += 1
    for bucket, count in sorted(buckets.items()):
        pct = count / len(ids) * 100
        print(f"  {bucket}: {count:,} ({pct:.1f}%)")

    print("\n=== Top titles in kept pool ===")
    title_counts = Counter(slim[i]["current_title"] for i in ids if i in slim)
    for title, count in title_counts.most_common(15):
        print(f"  {title}: {count:,}")

    print("\n=== Availability breakdown ===")
    avail_counts = Counter()
    for i in ids:
        notice = features[i]["notice_period_days"]
        if notice <= 30:
            avail_counts["≤30d notice"] += 1
        elif notice <= 60:
            avail_counts["31-60d notice"] += 1
        elif notice <= 90:
            avail_counts["61-90d notice"] += 1
        else:
            avail_counts[">90d notice"] += 1
    for bucket, count in sorted(avail_counts.items()):
        pct = count / len(ids) * 100
        print(f"  {bucket}: {count:,} ({pct:.1f}%)")

    print("\n=== Recommendations for weight tuning ===")
    mean_prod = statistics.mean(feat_vals["production_score"])
    mean_loc = statistics.mean(feat_vals["location_score"])
    mean_avail = statistics.mean(feat_vals["availability_score"])
    if mean_prod < 0.2:
        print("  ⚠ Production scores are low overall — consider increasing production_score weight in career_arc")
    if mean_loc < 0.5:
        print("  ⚠ Many candidates are outside Tier-1 India — location weight is a strong discriminator")
    if mean_avail < 0.4:
        print("  ⚠ Low overall availability — behavioral multiplier will significantly reorder rankings")
    print("  Done. Use these insights to adjust weights in src/config.py before final submission.")


if __name__ == "__main__":
    main()
