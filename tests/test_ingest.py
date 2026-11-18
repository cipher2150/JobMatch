from unittest.mock import MagicMock

import ingest


def job(id, body):
    return {"id": id, "source": "test", "title": "Engineer", "company": "Co", "location": "Anywhere",
            "url": "https://example.com", "body": body, "posted_at": None}


def test_unchanged_hash_skips_reembedding(monkeypatch):
    feed = [job("t:1", "same text"), job("t:2", "edited text"), job("t:3", "brand new")]
    monkeypatch.setattr(ingest, "SOURCES", [lambda: iter(feed)])
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        {"id": "t:1", "content_hash": ingest.content_hash(job("t:1", "same text"))},
        {"id": "t:2", "content_hash": "hash-of-the-old-text"},
    ]
    embedded = []
    monkeypatch.setattr(ingest.llm, "embed", lambda texts, task: embedded.extend(texts) or [[0.0]] * len(texts))

    summary = ingest.refresh(conn)

    assert embedded == ["Engineer\nedited text", "Engineer\nbrand new"]
    assert (summary["fetched"], summary["new"], summary["changed"], summary["reembedded"]) == (3, 1, 1, 2)
