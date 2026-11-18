import pytest
from pydantic import ValidationError

import match
from match import Fit

VALID = '{"skills": ["python"], "years_experience": 3, "current_titles": ["Dev"], "locations": [], "summary": "Backend dev."}'


def fake_llm(monkeypatch, *replies):
    calls = []
    monkeypatch.setattr(match.llm, "chat_json", lambda *a: calls.append(a) or replies[len(calls) - 1])
    return calls


def test_parser_rejects_malformed_json_after_one_retry(monkeypatch):
    calls = fake_llm(monkeypatch, "not json {", '{"skills": "python"}')
    with pytest.raises(ValidationError):
        match.parse_resume("resume text")
    assert len(calls) == 2


def test_parser_retry_recovers(monkeypatch):
    fake_llm(monkeypatch, "not json {", VALID)
    assert match.parse_resume("resume text").skills == ["python"]


def fit(job_id, skill, seniority, location):
    return Fit(job_id=job_id, skill_fit=skill, seniority_fit=seniority, location_fit=location, reason="")


def test_rank_orders_by_weighted_score():
    fits = [fit("a", 9, 2, 0),    # 5.4 + 0.5  + 0   = 5.9
            fit("b", 6, 10, 10),  # 3.6 + 2.5  + 1.5 = 7.6
            fit("c", 10, 10, 10),  # 10
            fit("d", 0, 10, 10),  # 0   + 2.5  + 1.5 = 4.0
            fit("b", 1, 1, 1)]    # repeated id: the lower duplicate is dropped
    assert [f.job_id for f in match.rank(fits)] == ["c", "b", "a", "d"]
    assert [f.job_id for f in match.rank(fits, n=2)] == ["c", "b"]
