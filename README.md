# Redrob AI Challenge — Candidate Ranker

## Architecture

Five-stage pipeline designed to rank like a senior technical recruiter, not a keyword matcher.

**Stage 1 — Hard filter + honeypot detection**
Eliminates non-technical roles, 100% services-career candidates, and synthetic honeypot profiles.
Honeypots detected by: career timeline inconsistency, advanced-claim vs assessment-score mismatch,
identical copy-pasted descriptions, and title-vs-career mismatch (Java Developer with 100% React work).
Conditional titles (Frontend, DevOps, Java) pass only if career history shows real ML evidence.

**Stage 2 — BGE bi-encoder retrieval**
Career descriptions embedded with BAAI/bge-small-en-v1.5 (IR-optimised, 33MB, fast CPU).
JD embedded from two angles: technical requirements + career narrative.
Cosine similarity retrieves top-500 in ~0.4 seconds.

**Stage 3 — Cross-encoder re-ranking**
ms-marco-MiniLM-L-6-v2 scores each (JD, candidate) pair directly.
Top-500 re-ranked to top-200 with significantly higher precision than bi-encoder alone.

**Stage 4 — XGBoost LambdaMART (Learning-to-Rank)**
Trained offline on 24 features using rank:ndcg objective — the same metric used for judging.
Key features unique to this solution:
- skill.duration_months: how long they actually used each skill (not just claimed it)
- skill.endorsements: social proof per skill
- product_industry_score: Swiggy/Zomato/Uber experience vs generic IT services
- search_appearance_30d: how many recruiters are already searching for this candidate
- assessment_credibility: cross-validates advanced claims against actual test scores
- pre_llm_score: months of ML experience before 2022 (filters LangChain tourists)

**Stage 5 — Behavioral multiplier**
Multiplicative (not additive) so inactive/unresponsive candidates cannot rank in top-10
regardless of skill match. Combines recency, notice period, recruiter response rate,
open-to-work status, and engagement signals.

**Reasoning — LLM-generated, cached offline**
Anthropic API called during precompute for top-300 candidates.
Responses cached to artifacts/reasoning_cache.json.
At ranking time: zero API calls, instant cache lookup.
Result: genuinely specific recruiter-quality reasoning, not templates.

## Reproduce

```bash
pip install -r requirements.txt

# Step 1: precompute (offline, no time limit)
python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --out artifacts/

# Step 2: rank (completes in ~2 minutes on CPU)
python rank.py --candidates data/candidates.jsonl --out submission.csv

# Step 3: validate
python validate_submission.py submission.csv
```

If no internet during precompute, add `--skip-reasoning` to step 1.
Ranking still works — template reasoning is used as fallback.

## Sandbox demo

```bash
streamlit run app.py
```

## Project structure

```
├── rank.py                          main ranker (≤5min, CPU, no network)
├── app.py                           streamlit sandbox demo
├── src/
│   ├── config.py                    all weights, thresholds, JD text
│   ├── filters.py                   hard filters + honeypot detection
│   ├── features.py                  24 features per candidate
│   ├── ltr.py                       XGBoost LambdaMART model
│   ├── text_builder.py              candidate → embedding text
│   └── reasoning.py                 LLM reasoning + template fallback
├── scripts/
│   └── precompute_embeddings.py     offline: embed + train LTR + generate reasoning
└── tests/
    └── test_pipeline.py             end-to-end tests on sample data
```
