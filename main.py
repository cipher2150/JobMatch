import hmac
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import Cookie, FastAPI, Header, HTTPException, Request
from fastapi.templating import Jinja2Templates
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

import ingest
import match

ROOT = Path(__file__).parent
templates = Jinja2Templates(ROOT / "templates")


def db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


@asynccontextmanager
async def lifespan(app):
    # DDL goes over the direct URL; the -pooler URL (PgBouncer, transaction mode) is for app traffic
    with psycopg.connect(os.environ.get("DATABASE_URL_UNPOOLED") or os.environ["DATABASE_URL"]) as conn:
        conn.execute((ROOT / "schema.sql").read_text())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index(request: Request, user_key: str | None = Cookie(None)):
    resp = templates.TemplateResponse(request, "index.html")
    if not user_key:
        resp.set_cookie("user_key", uuid.uuid4().hex, max_age=365 * 86400, httponly=True, samesite="lax")
    return resp


class MatchIn(BaseModel):
    resume: str = Field(min_length=100, max_length=20_000)  # every /match costs ~11 LLM calls


@app.post("/match")
def post_match(body: MatchIn, rerank: bool = True, user_key: str | None = Cookie(None)):
    with db() as conn:
        return match.match(conn, body.resume, user_key, rerank)


class FeedbackIn(BaseModel):
    job_id: str
    verdict: Literal["up", "down"]


@app.post("/feedback", status_code=204)
def feedback(body: FeedbackIn, user_key: str | None = Cookie(None)):
    if not user_key:
        raise HTTPException(400, "missing user_key cookie; load / first")
    with db() as conn:
        try:
            conn.execute("insert into feedback (user_key, job_id, verdict) values (%s, %s, %s)",
                         (user_key, body.job_id, body.verdict))
        except psycopg.errors.ForeignKeyViolation:
            raise HTTPException(404, "unknown job_id")


@app.post("/admin/refresh")
def admin_refresh(x_admin_token: str = Header("")):
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected or not hmac.compare_digest(x_admin_token.encode(), expected.encode()):
        raise HTTPException(401)
    with db() as conn:
        return ingest.refresh(conn)
