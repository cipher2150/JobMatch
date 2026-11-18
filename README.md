# JobMatch

Paste a resume, get the 10 best-matching remote jobs (from [Jobicy](https://jobicy.com)), each with a one-line reason.

## What it gets wrong

Observed while building, 2026-11-18. Anything marked **pending** needs a real run and stays blank rather than guessed.

1. **The source the spec named is nearly empty.** Remotive's public API returned `"total-job-count": 16`. Top-200 retrieval over 16 rows does nothing and precision@10 would be noise. *Fix applied:* ingest from Jobicy (cursor-paged, ~100 per page). A second source is one function added to `ingest.SOURCES`.
2. **The job pool is skewed toward US and senior roles.** Of the first 100 Jobicy postings, 33 were USA-only and 7 open to "Anywhere". By level, 37 were Senior, 28 Director and 35 "Any", with none labelled junior. Junior and non-US resumes (eval cases 1, 9, 21, 24) have few jobs they can apply to, so their precision@10 is capped by the data, not the ranker. Measured effect: **pending**. *Fix:* add a source with entry-level and worldwide roles, or report precision only over cases the pool can satisfy.
3. **Postings drop out after 7 days.** Jobicy serves only the last 7 days, so a job that is still open on day 8 gets `active = false`. *Fix:* Jobicy's `/api/jobs/status?ids=` reports which listings are closed; deactivate based on that instead of on absence from the feed.
4. **pending:** reranker, parser and judge failure modes from the first full eval run.

## Architecture

```
GitHub Actions, every 6h ─ POST /admin/refresh ─▶ ingest.py ◀── Jobicy API
                                                    │  sha256(title+body) unchanged → no embedding call
                                                    ▼
                                      Neon Postgres + pgvector (jobs, feedback)
                                                    ▲
Browser ─ POST /match ─▶ match.py
   1. parse resume → Profile JSON          (1 Gemini call, Pydantic-validated, 1 retry)
   2. embed summary + skills → 200 nearest active jobs (cosine, HNSW)
   3. rerank in 10 batches of 20, in parallel (1 Gemini call each; up to 3 liked + 3 disliked titles as taste)
   4. score = 0.6·skill + 0.25·seniority + 0.15·location → top 10
```

| File | Role |
|---|---|
| `main.py` | FastAPI routes: `/`, `/match`, `/feedback`, `/admin/refresh`; applies `schema.sql` on startup |
| `ingest.py` | sources, HTML→text, hash-gated upsert, deactivation |
| `match.py` | parse → retrieve → rerank → rank |
| `llm.py` | the only module that talks to Gemini |
| `prompts.py` | every prompt; its hash is logged with each eval run |
| `evals/` | 30 cases, `run.py`, `history.csv` |

## Precision@10

A hit is a returned title containing any of the case's `should_rank_high` keywords. `low@10` counts `should_rank_low` titles (lower is better).

| Stage | precision@10 | low@10 | reason judge (0–5) |
|---|---|---|---|
| Retrieval only (`?rerank=false`) | pending | pending | n/a |
| + LLM reranker | pending | pending | pending |

## Run it

The database is Neon project `jobMatch`, branch `production` (linked in `.neon`, declared in `neon.ts`). `neon env pull` writes `DATABASE_URL` (pooled, for the app) and `DATABASE_URL_UNPOOLED` (direct, for the startup DDL) into `.env`; add `GEMINI_API_KEY` and `ADMIN_TOKEN` there yourself.

```sh
neon env pull                                     # Neon vars → .env
set -a; . ./.env; set +a                          # load .env into this shell
docker compose up --build                         # or: .venv/bin/uvicorn main:app
curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" localhost:8000/admin/refresh   # first load
open http://localhost:8000
.venv/bin/python evals/run.py                     # with the app running; appends to evals/history.csv
.venv/bin/pytest
```

Deploy on Fly: `fly launch --no-deploy` (keep `fly.toml`, change `app`), then `fly secrets set DATABASE_URL=… DATABASE_URL_UNPOOLED=… GEMINI_API_KEY=… ADMIN_TOKEN=…` and `fly deploy`. For the cron, add the repo secrets `APP_URL` and `ADMIN_TOKEN`.

## Deviations from the brief

- **Jobicy instead of Remotive:** see failure mode 1.
- **Gemini instead of OpenAI.** Embeddings come from `gemini-embedding-001` truncated to 1536 dims, so the schema keeps `vector(1536)`. The chat model is `GEMINI_MODEL` (default `gemini-2.5-flash`).
- **`/match` takes JSON** `{"resume": "…"}`, capped at 20k characters, instead of a raw text body.
- **No RRF test:** there's only one ranked list, so the test covers weighted-score ordering instead.
- **Partial HNSW index** (`where active`) and an idempotent `schema.sql` that the app applies on startup.

## Costs and limits

- One `/match` makes 11 LLM calls plus 1 embedding call. One eval run makes about 390 LLM calls (30 cases × (2 parses + 10 rerank + 1 judge)). Free-tier Gemini keys may hit their daily cap partway through an eval run. The client retries 429s, but a daily cap is a hard stop.
- `/match` is public and unauthenticated, so anyone with the URL spends your quota. Set a spend cap on the key.
- Jobicy terms: credit Jobicy and link to the listing (the page does both), and poll at most hourly (the cron runs every 6h).
