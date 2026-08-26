"""Copilot: natural-language questions over the contact database.

Two modes, degrading gracefully offline:

* **LLM mode** — if ``SAGAR_LLM_ENDPOINT`` points at an OpenAI-compatible
  chat endpoint (e.g. a local Ollama), the question plus the table schema is
  sent there and the returned SQL is validated (SELECT-only) and executed.
* **Rule mode** (always available, the offline fallback) — a pattern grammar
  covering the questions surveyors actually ask: counts, top-N by severity or
  confidence, class filters, severity bands, review status, per-survey
  filters, and proximity to named sensitive layers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from api.db import ContactRepo
from geoscribe.severity import load_layers
from tridentnet.classes import REPORTABLE

REPO_ROOT = Path(__file__).resolve().parents[1]

_CLASS_SYNONYMS: dict[str, str] = {
    "ghost net": "ghost_net",
    "ghostnet": "ghost_net",
    "net": "ghost_net",
    "wreck": "wreck",
    "shipwreck": "wreck",
    "ship": "wreck",
    "aircraft": "aircraft",
    "plane": "aircraft",
    "pipeline": "pipeline",
    "pipe": "pipeline",
    "drum": "cylinder_drum",
    "cylinder": "cylinder_drum",
    "barrel": "cylinder_drum",
    "tire": "tire",
    "tyre": "tire",
    "container": "container",
    "body": "human_body",
    "victim": "human_body",
    "mine": "mine_like",
}

_SEVERITY_WORDS = {"critical": 75.0, "severe": 75.0, "high": 50.0, "medium": 25.0}
_LAYER_WORDS = {
    "turtle": "turtle_nesting_zone",
    "nesting": "turtle_nesting_zone",
    "shipping": "shipping_lane",
    "lane": "shipping_lane",
    "protected": "marine_protected_area",
    "mpa": "marine_protected_area",
}


def _find_class(question: str) -> str | None:
    for phrase, cls in _CLASS_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}s?\b", question):
            return cls
    for cls in REPORTABLE:
        if cls.replace("_", " ") in question:
            return cls
    return None


def _try_llm(question: str, repo: ContactRepo) -> dict[str, Any] | None:
    endpoint = os.environ.get("SAGAR_LLM_ENDPOINT")
    if not endpoint:
        return None
    try:
        import json
        import urllib.request

        schema = (
            "Table contacts(id TEXT, survey TEXT, cls TEXT, confidence REAL /*0-100*/, "
            "severity REAL /*0-100*/, lat REAL, lon REAL, review TEXT "
            "/*pending|confirmed|rejected*/, detected_at TEXT). SQLite dialect."
        )
        body = {
            "model": os.environ.get("SAGAR_LLM_MODEL", "llama3.2"),
            "messages": [
                {
                    "role": "system",
                    "content": "Translate the question to ONE SQLite SELECT statement over: "
                    + schema
                    + " Reply with the SQL only, no prose, no code fences.",
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as resp:
            reply = json.loads(resp.read())
        sql = reply["choices"][0]["message"]["content"].strip().strip("`;")
        rows = repo.run_sql(sql)  # raises unless SELECT
        return {"mode": "llm", "sql": sql, "rows": rows, "answer": _summarize_rows(rows)}
    except Exception:  # noqa: BLE001 - any LLM failure falls back to rules
        return None


def _summarize_rows(rows: list[dict]) -> str:
    if not rows:
        return "No matching contacts."
    if len(rows) == 1 and len(rows[0]) == 1:
        value = next(iter(rows[0].values()))
        return f"Result: {value}"
    return f"{len(rows)} rows returned."


def ask(question: str, repo: ContactRepo, layer_dir: str | Path | None = None) -> dict[str, Any]:
    """Answer a natural-language question; never raises, always answers."""
    q = question.strip().lower()
    llm = _try_llm(q, repo)
    if llm is not None:
        return llm

    clauses: list[str] = []
    params: list[Any] = []
    cls = _find_class(q)
    if cls:
        clauses.append("cls = ?")
        params.append(cls)
    # "most severe" / "highest severity" are orderings, not thresholds — strip
    # them before scanning for band words like "severe" or "high".
    q_thresholds = re.sub(
        r"\b(?:most|least|highest|lowest)\s+(?:severe|severity|critical|high)\b", "", q
    )
    for word, threshold in _SEVERITY_WORDS.items():
        if word in q_thresholds:
            clauses.append("severity >= ?")
            params.append(threshold)
            break
    for status in ("confirmed", "rejected", "pending"):
        if status in q:
            clauses.append("review = ?")
            params.append(status)
            break
    survey_match = re.search(r"survey\s+([\w.\-]+)", q)
    if survey_match:
        clauses.append("survey LIKE ?")
        params.append(f"%{survey_match.group(1)}%")

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    # Proximity questions run through the severity layer geometry in Python.
    layer_key = next((v for k, v in _LAYER_WORDS.items() if k in q), None)
    if layer_key:
        layers = load_layers(layer_dir or REPO_ROOT / "data" / "layers")
        layer = next((la for la in layers if la.name == layer_key), None)
        rows = repo.run_sql(f"SELECT json FROM contacts{where}", tuple(params))
        near = []
        if layer is not None:
            from geoscribe.severity import _distance_to_geometry_m

            for row in rows:
                c = row["json"]
                dist = min(
                    (
                        _distance_to_geometry_m(c["lon"], c["lat"], f.get("geometry") or {})
                        for f in layer.features
                    ),
                    default=float("inf"),
                )
                if dist <= 2000.0:
                    near.append({"id": c["id"], "cls": c["cls"], "distance_m": round(dist, 1)})
        answer = (
            f"{len(near)} contact(s) within 2 km of {layer_key.replace('_', ' ')}: "
            + ", ".join(f"{n['id']} ({n['cls']}, {n['distance_m']} m)" for n in near)
            if near
            else f"No contacts within 2 km of {layer_key.replace('_', ' ')}."
        )
        return {"mode": "rules", "sql": None, "rows": near, "answer": answer}

    count_match = re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q)
    top_match = re.search(r"\b(?:top|first)\s*(\d+)?\b|\bhighest\b|\bmost severe\b|\bworst\b", q)

    if count_match:
        sql = f"SELECT COUNT(*) AS n FROM contacts{where}"
        rows = repo.run_sql(sql, tuple(params))
        n = rows[0]["n"]
        subject = cls.replace("_", " ") + " contacts" if cls else "contacts"
        return {"mode": "rules", "sql": sql, "rows": rows, "answer": f"{n} {subject} match."}

    limit = 5
    if top_match and top_match.group(1):
        limit = int(top_match.group(1))
    order = "confidence" if "confiden" in q else "severity"
    sql = (
        f"SELECT id, cls, confidence, severity, lat, lon, review FROM contacts{where} "
        f"ORDER BY {order} DESC LIMIT ?"
    )
    rows = repo.run_sql(sql, (*params, limit))
    if not rows:
        return {"mode": "rules", "sql": sql, "rows": [], "answer": "No matching contacts."}
    lines = [
        f"{r['id']}: {r['cls']} (severity {r['severity']}, confidence {r['confidence']}%)"
        for r in rows
    ]
    return {
        "mode": "rules",
        "sql": sql,
        "rows": rows,
        "answer": f"Top {len(rows)} by {order}:\n" + "\n".join(lines),
    }
