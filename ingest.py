import hashlib
import html
import re
import time

import httpx

import llm


def html_to_text(s: str | None) -> str:
    # shortcut: regex tag strip is enough for job-board HTML; use a real parser if <script>/<style> show up
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def fetch_jobicy():
    """Jobicy public API: jobs from the last 7 days, 100 per page, cursor-paged. Poll at most hourly."""
    params = {"count": 100}
    for _ in range(50):  # cap guards against a cursor that never ends
        d = httpx.get("https://jobicy.com/api/jobs", params=params, timeout=60,
                      headers={"User-Agent": "jobmatch-mvp"}).raise_for_status().json()
        for j in d["jobs"]:
            industry = j["jobIndustry"] if isinstance(j["jobIndustry"], list) else [j["jobIndustry"]]
            yield {
                "id": f"jobicy:{j['id']}", "source": "jobicy",
                "title": html.unescape(j["jobTitle"]), "company": html.unescape(j["companyName"]),
                "location": j["jobGeo"], "url": j["url"], "posted_at": j["pubDate"],
                # level and industry are fields, not prose; prepend them so the embedding and reranker see them
                "body": f"Level: {j['jobLevel']}. Industry: {html.unescape(', '.join(industry))}. "
                        + html_to_text(j["jobDescription"]),
            }
        if not d.get("hasMore"):
            return
        params["cursor"] = d["nextCursor"]


# Adding a source = one fetch_x() yielding dicts shaped like fetch_jobicy's, appended here.
SOURCES = [fetch_jobicy]

UPSERT = """
insert into jobs (id, source, title, company, location, url, body, content_hash, embedding, posted_at)
values (%(id)s, %(source)s, %(title)s, %(company)s, %(location)s, %(url)s, %(body)s, %(content_hash)s,
        %(embedding)s::vector, %(posted_at)s)
on conflict (id) do update set
  title = excluded.title, company = excluded.company, location = excluded.location, url = excluded.url,
  body = excluded.body, content_hash = excluded.content_hash, embedding = excluded.embedding,
  posted_at = excluded.posted_at, last_seen = now(), active = true
"""


def content_hash(job: dict) -> str:
    return hashlib.sha256(f"{job['title']}\n{job['body']}".encode()).hexdigest()


def refresh(conn) -> dict:
    t0 = time.time()
    feed = {j["id"]: j for source in SOURCES for j in source()}
    if not feed:
        raise RuntimeError("feed returned 0 jobs; refusing to deactivate everything")
    for j in feed.values():
        j["content_hash"] = content_hash(j)
    known = {r["id"]: r["content_hash"] for r in conn.execute("select id, content_hash from jobs").fetchall()}

    # The point of the project: only jobs whose text is new or changed cost an embedding call.
    stale = [j for j in feed.values() if known.get(j["id"]) != j["content_hash"]]
    unchanged = [i for i, j in feed.items() if known.get(i) == j["content_hash"]]

    conn.execute("update jobs set last_seen = now(), active = true where id = any(%s)", (unchanged,))
    deactivated = conn.execute("update jobs set active = false where active and id <> all(%s)", (list(feed),)).rowcount
    conn.commit()
    for i in range(0, len(stale), 50):
        chunk = stale[i:i + 50]
        # shortcut: embeds the first 6000 chars (~1.5k tokens, under the model's 2048-token input cap)
        vectors = llm.embed([f"{j['title']}\n{j['body']}"[:6000] for j in chunk], "RETRIEVAL_DOCUMENT")
        with conn.cursor() as cur:
            cur.executemany(UPSERT, [{**j, "embedding": str(v)} for j, v in zip(chunk, vectors)])
        conn.commit()  # keep finished chunks if a later one hits a rate limit; the next refresh resumes

    new = sum(j["id"] not in known for j in stale)
    return {"fetched": len(feed), "new": new, "changed": len(stale) - new, "reembedded": len(stale),
            "deactivated": deactivated, "seconds": round(time.time() - t0, 1)}
