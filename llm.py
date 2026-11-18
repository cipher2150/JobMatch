"""The only module that talks to the model provider (Gemini). Swap providers here."""
import os
from functools import cache

from google import genai
from google.genai import types

CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = "gemini-embedding-001"
DIMS = 1536  # must match vector(1536) in schema.sql; pgvector's HNSW index caps at 2000 dims


@cache
def _client():
    # Retries 429/5xx with backoff: free-tier Gemini allows only a few requests per minute.
    return genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=6)))


def chat_json(system: str, user: str, schema) -> str:
    """One call constrained to `schema` (a Pydantic model). Returns raw text; callers validate."""
    r = _client().models.generate_content(
        model=CHAT_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            response_mime_type="application/json", response_schema=schema,
        ),
    )
    return r.text


def embed(texts: list[str], task: str) -> list[list[float]]:
    """task: RETRIEVAL_DOCUMENT for jobs, RETRIEVAL_QUERY for resumes. Max 100 texts per call."""
    r = _client().models.embed_content(
        model=EMBED_MODEL, contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIMS),
    )
    return [e.values for e in r.embeddings]
