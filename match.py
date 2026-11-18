import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, ValidationError

import llm
import prompts

RETRIEVE_K, BATCH = 200, 20
WEIGHTS = {"skill_fit": 0.6, "seniority_fit": 0.25, "location_fit": 0.15}


class Profile(BaseModel):
    skills: list[str]
    years_experience: float
    current_titles: list[str]
    locations: list[str]
    summary: str


class Fit(BaseModel):
    job_id: str
    skill_fit: int = Field(ge=0, le=10)
    seniority_fit: int = Field(ge=0, le=10)
    location_fit: int = Field(ge=0, le=10)
    reason: str


class Fits(BaseModel):
    results: list[Fit]


def ask(system: str, user: str, model):
    """One LLM call validated by `model`; retries once if the JSON is malformed or off-schema."""
    try:
        return model.model_validate_json(llm.chat_json(system, user, model))
    except ValidationError:
        return model.model_validate_json(llm.chat_json(system, user, model))


def parse_resume(text: str) -> Profile:
    return ask(prompts.PARSE, text, Profile)


def score(fit: Fit) -> float:
    return sum(getattr(fit, k) * w for k, w in WEIGHTS.items())


def rank(fits: list[Fit], n: int = 10) -> list[Fit]:
    best = {}
    for f in sorted(fits, key=score, reverse=True):
        best.setdefault(f.job_id, f)  # the model sometimes repeats a job; keep its best score
    return list(best.values())[:n]


def rerank_batch(profile: Profile, jobs: list[dict], taste: dict) -> list[Fit]:
    payload = {"candidate": profile.model_dump(),
               "jobs": [{"job_id": j["job_id"], "title": j["title"], "company": j["company"],
                         "location": j["location"], "description": j["body"][:1500]} for j in jobs]}
    if any(taste.values()):
        payload["taste"] = taste
    ids = {j["job_id"] for j in jobs}
    return [f for f in ask(prompts.RERANK, json.dumps(payload), Fits).results if f.job_id in ids]


TASTE_SQL = """select j.title from feedback f join jobs j on j.id = f.job_id
where f.user_key = %s and f.verdict = %s order by f.created_at desc limit 3"""

RETRIEVE_SQL = """select id as job_id, title, company, location, url, body, 1 - (embedding <=> %(v)s::vector) as similarity
from jobs where active order by embedding <=> %(v)s::vector limit %(k)s"""


def card(job: dict, score: float, reason: str) -> dict:
    return {k: job[k] for k in ("job_id", "title", "company", "location", "url")} | {"score": round(score, 1), "reason": reason}


def match(conn, resume: str, user_key: str | None, rerank: bool = True) -> list[dict]:
    profile = parse_resume(resume)
    vec = llm.embed([profile.summary + "\nSkills: " + ", ".join(profile.skills)], "RETRIEVAL_QUERY")[0]
    # default 40 would silently cap LIMIT 200 at ~40 rows; LOCAL so it can't leak to other pooled clients
    conn.execute("set local hnsw.ef_search = 400")
    jobs = conn.execute(RETRIEVE_SQL, {"v": str(vec), "k": RETRIEVE_K if rerank else 10}).fetchall()
    if not rerank:
        return [card(j, j["similarity"] * 10, "") for j in jobs]

    taste = {v: [r["title"] for r in conn.execute(TASTE_SQL, (user_key, v))] for v in ("up", "down")} if user_key else {}
    batches = [jobs[i:i + BATCH] for i in range(0, len(jobs), BATCH)]
    with ThreadPoolExecutor(10) as ex:
        fits = [f for batch in ex.map(lambda b: rerank_batch(profile, b, taste), batches) for f in batch]
    by_id = {j["job_id"]: j for j in jobs}
    return [card(by_id[f.job_id], score(f), f.reason) for f in rank(fits)]
