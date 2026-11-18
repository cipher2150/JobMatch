"""Run every case through /match, log precision@10 and an LLM-judge score. Usage: python evals/run.py

Needs the app running (EVAL_URL, default http://localhost:8000) and GEMINI_API_KEY for the judge.
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import prompts  # noqa: E402
from match import ask  # noqa: E402

URL = os.environ.get("EVAL_URL", "http://localhost:8000")
HISTORY = HERE / "history.csv"
METRICS = ["p10_retrieval", "p10_rerank", "low10_retrieval", "low10_rerank", "judge"]
FIELDS = ["timestamp", "prompts_sha", "cases", "errors"] + METRICS


class Judgement(BaseModel):
    score: float = Field(ge=0, le=5)
    why: str


def share(jobs: list[dict], keywords: list[str]) -> float:
    """Fraction of the 10 slots whose title contains any keyword; missing slots count as misses."""
    return sum(any(k.lower() in j["title"].lower() for k in keywords) for j in jobs[:10]) / 10


def run_case(case: dict) -> dict:
    out = {}
    for mode in ("retrieval", "rerank"):
        jobs = httpx.post(f"{URL}/match", params={"rerank": mode == "rerank"}, json={"resume": case["resume"]},
                          timeout=900).raise_for_status().json()
        out[f"p10_{mode}"] = share(jobs, case["should_rank_high"])
        out[f"low10_{mode}"] = share(jobs, case["should_rank_low"])
    results = [{k: j[k] for k in ("title", "company", "reason")} for j in jobs]
    out["judge"] = ask(prompts.JUDGE, json.dumps({"resume": case["resume"], "results": results}), Judgement).score
    return out


def main():
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        try:
            r = run_case(case)
        except Exception as e:  # one flaky case shouldn't sink a 30-case run; it shows up in `errors`
            print(f"case {i:2}: ERROR {e!r}")
            continue
        rows.append(r)
        print(f"case {i:2}: " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()))

    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "prompts_sha": hashlib.sha256((HERE.parent / "prompts.py").read_bytes()).hexdigest()[:12],
              "cases": len(cases), "errors": len(cases) - len(rows)}
    if rows:
        record |= {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}

    prev = list(csv.DictReader(HISTORY.open()))[-1] if HISTORY.exists() else {}
    with HISTORY.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="")
        if not prev:
            w.writeheader()
        w.writerow(record)

    print(f"\nprompts {record['prompts_sha']}" + (f" (was {prev['prompts_sha']})" if prev.get("prompts_sha") not in (None, record["prompts_sha"]) else ""))
    for k in ["errors"] + METRICS:
        now, before = record.get(k, ""), prev.get(k, "")
        delta = f"  ({float(now) - float(before):+.3f} vs previous)" if now != "" and before != "" else ""
        print(f"{k:>16}: {now}{delta}")
    sys.exit(1 if not rows else 0)


if __name__ == "__main__":
    main()
