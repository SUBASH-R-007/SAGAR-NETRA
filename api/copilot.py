"""Copilot: natural-language questions over the contact database.

Two modes, degrading gracefully offline:

* **LLM mode** — if ``SAGAR_LLM_ENDPOINT`` points at an OpenAI-compatible
  chat endpoint (e.g. a local Ollama), the question plus the table schema is
  sent there and the returned SQL is validated (SELECT-only) and executed.
* **Rule mode** (always available, the offline fallback) — a pattern grammar
  covering the questions surveyors actually ask: counts, top-N by severity or
  confidence, class filters, severity bands, review status, per-survey
  filters, proximity to named sensitive layers, and dimension/depth filters
  ("longer than 5 m", "over 2 m tall", "between 20 and 40 m depth").

Dimension and depth filters run in Python over the stored contact JSON after
the SQL fetch: length/height live inside the nested ``dims`` document and are
not indexed columns, and a contact with no shadow has ``height_m: None`` —
"unmeasured" must never satisfy a size question, a distinction SQL NULL
comparisons get wrong in both directions. Every rule-mode answer states the
filters it applied so the operator can audit what was actually asked.

``draft_summary`` composes a survey-level briefing (counts by class, worst
contacts with positions, physics violations, review progress, nearest
sensitive layer). The rule-based markdown template always works offline; when
the LLM endpoint is configured it merely polishes the wording — numbers stay
authoritative from SQL, never from the model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
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

_SUMMARY_RE = re.compile(r"\b(?:summary|summarize|summarise|report)\b")

_NUM = r"(\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class DimFilter:
    """One dimension/depth predicate parsed from the question.

    ``field`` is ``length``/``height`` (contact ``dims``, metres) or ``depth``
    (water depth at the contact, metres). Bounds are open (``>``/``<``) for
    "-er than" phrasings and closed for "at least"/"between" ranges.
    """

    field: str  # "length" | "height" | "depth"
    lo: float | None = None
    hi: float | None = None
    inclusive: bool = False

    def value_of(self, contact: dict[str, Any]) -> float | None:
        if self.field == "depth":
            value = contact.get("depth_m")
        else:
            value = (contact.get("dims") or {}).get(f"{self.field}_m")
        return None if value is None else float(value)

    def matches(self, contact: dict[str, Any]) -> bool:
        """True when the contact provably satisfies the predicate.

        A missing measurement (no shadow -> ``height_m`` is None; no altitude
        -> ``depth_m`` is None) never matches: the copilot must not claim an
        unmeasured contact is "over 2 m tall".
        """
        value = self.value_of(contact)
        if value is None:
            return False
        if self.lo is not None and not (value >= self.lo if self.inclusive else value > self.lo):
            return False
        if self.hi is not None and not (value <= self.hi if self.inclusive else value < self.hi):
            return False
        return True

    def describe(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"{self.field} {self.lo:g}-{self.hi:g} m"
        if self.lo is not None:
            return f"{self.field} {'>=' if self.inclusive else '>'} {self.lo:g} m"
        return f"{self.field} {'<=' if self.inclusive else '<'} {self.hi:g} m"


#: (pattern, field, kind, inclusive): kind "lo"/"hi" reads one number, "range"
#: two. Depth ranges require a depth word so "between 5 and 10 m long" can
#: never be misread as a depth band.
_DIM_PATTERNS: tuple[tuple[re.Pattern[str], str, str, bool], ...] = (
    (re.compile(rf"\blonger than {_NUM}\s*m\b"), "length", "lo", False),
    (re.compile(rf"\bshorter than {_NUM}\s*m\b"), "length", "hi", False),
    (re.compile(rf"\b(?:over|more than) {_NUM}\s*m\s+long\b"), "length", "lo", False),
    (re.compile(rf"\bat least {_NUM}\s*m\s+long\b"), "length", "lo", True),
    (re.compile(rf"\b(?:under|less than) {_NUM}\s*m\s+long\b"), "length", "hi", False),
    (
        re.compile(rf"\bbetween {_NUM}\s*(?:m\s+)?and {_NUM}\s*m\s+long\b"),
        "length", "range", True,
    ),
    (re.compile(rf"\b(?:taller|higher) than {_NUM}\s*m\b"), "height", "lo", False),
    (re.compile(rf"\b(?:over|more than) {_NUM}\s*m\s+(?:tall|high)\b"), "height", "lo", False),
    (re.compile(rf"\bat least {_NUM}\s*m\s+(?:tall|high)\b"), "height", "lo", True),
    (re.compile(rf"\b(?:under|less than) {_NUM}\s*m\s+(?:tall|high)\b"), "height", "hi", False),
    (
        re.compile(
            rf"\bbetween {_NUM}\s*(?:m\s+)?and {_NUM}\s*m"
            r"(?:\s+(?:of\s+)?(?:depth|deep|water(?:\s+depth)?))\b"
        ),
        "depth", "range", True,
    ),
    (re.compile(rf"\bdeeper than {_NUM}\s*m\b"), "depth", "lo", False),
    (re.compile(rf"\bshallower than {_NUM}\s*m\b"), "depth", "hi", False),
)


def _parse_dim_filters(q: str) -> tuple[list[DimFilter], str]:
    """Extract dimension/depth filters from *q*.

    Returns ``(filters, scrubbed_q)`` where the matched phrases are blanked
    out of the scrubbed question — measurement words like "high" or "deep"
    must not be re-parsed downstream as severity-band keywords ("over 2 m
    high" is a height filter, not a request for severity >= 50).
    """
    filters: list[DimFilter] = []
    scrubbed = q
    for pattern, fld, kind, inclusive in _DIM_PATTERNS:
        for match in pattern.finditer(q):
            if kind == "range":
                a, b = float(match.group(1)), float(match.group(2))
                filters.append(DimFilter(fld, lo=min(a, b), hi=max(a, b), inclusive=True))
            elif kind == "lo":
                filters.append(DimFilter(fld, lo=float(match.group(1)), inclusive=inclusive))
            else:
                filters.append(DimFilter(fld, hi=float(match.group(1)), inclusive=inclusive))
            scrubbed = scrubbed.replace(match.group(0), " ")
    return filters, scrubbed


def _passes(contact: dict[str, Any], filters: list[DimFilter]) -> bool:
    return all(f.matches(contact) for f in filters)


def _filters_note(applied: list[str]) -> str:
    return f" Filters: {'; '.join(applied)}." if applied else ""


def _find_class(question: str) -> str | None:
    for phrase, cls in _CLASS_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}s?\b", question):
            return cls
    for cls in REPORTABLE:
        if cls.replace("_", " ") in question:
            return cls
    return None


def _llm_chat(system: str, user: str, timeout_s: float = 20.0) -> str | None:
    """One chat completion against ``SAGAR_LLM_ENDPOINT``; None on any failure.

    The copilot must keep answering on a fully offline survey vessel, so every
    LLM interaction is optional and every failure mode (no endpoint, network
    down, malformed reply) collapses silently to None for the rule fallback.
    """
    endpoint = os.environ.get("SAGAR_LLM_ENDPOINT")
    if not endpoint:
        return None
    try:
        import json
        import urllib.request

        body = {
            "model": os.environ.get("SAGAR_LLM_MODEL", "llama3.2"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            reply = json.loads(resp.read())
        content = reply["choices"][0]["message"]["content"]
        return str(content).strip() or None
    except Exception:  # noqa: BLE001 - any LLM failure falls back to rules
        return None


def _try_llm(question: str, repo: ContactRepo) -> dict[str, Any] | None:
    schema = (
        "Table contacts(id TEXT, survey TEXT, cls TEXT, confidence REAL /*0-100*/, "
        "severity REAL /*0-100*/, lat REAL, lon REAL, review TEXT "
        "/*pending|confirmed|rejected*/, detected_at TEXT). SQLite dialect."
    )
    content = _llm_chat(
        "Translate the question to ONE SQLite SELECT statement over: "
        + schema
        + " Reply with the SQL only, no prose, no code fences.",
        question,
    )
    if content is None:
        return None
    sql = content.strip().strip("`;")
    try:
        rows = repo.run_sql(sql)  # raises unless SELECT
    except Exception:  # noqa: BLE001 - bad model SQL falls back to rules
        return None
    return {"mode": "llm", "sql": sql, "rows": rows, "answer": _summarize_rows(rows)}


def _summarize_rows(rows: list[dict]) -> str:
    if not rows:
        return "No matching contacts."
    if len(rows) == 1 and len(rows[0]) == 1:
        value = next(iter(rows[0].values()))
        return f"Result: {value}"
    return f"{len(rows)} rows returned."


def draft_summary(repo: ContactRepo, survey: str | None = None) -> dict[str, Any]:
    """Draft a survey summary briefing; ``{"answer": markdown, "mode": ...}``.

    Sections: contact count by class, the top-3 contacts by severity with
    positions, the physics-violation count (PhysiCheck flags that survived to
    the report — each one is a contact whose acoustics do not add up and needs
    eyes), review progress, and the closest approach to a sensitive layer.

    The rule-based markdown template is always produced from SQL truth; when
    ``SAGAR_LLM_ENDPOINT`` is set the LLM is asked to polish the prose (mode
    ``"llm"``), silently falling back to the template on any failure (mode
    ``"rules"``). Numbers therefore never depend on the model.
    """
    where, params = "", ()
    scope = "all surveys"
    if survey:
        where, params = " WHERE survey LIKE ?", (f"%{survey}%",)
        scope = f"survey {survey}"

    docs = [r["json"] for r in repo.run_sql(f"SELECT json FROM contacts{where}", params)]
    if not docs:
        return {"answer": f"No contacts found for {scope}.", "mode": "rules"}

    by_class: dict[str, int] = {}
    reviews = {"confirmed": 0, "rejected": 0, "pending": 0}
    violations = 0
    nearest: tuple[str, float, str] | None = None
    for c in docs:
        by_class[c["cls"]] = by_class.get(c["cls"], 0) + 1
        status = c.get("review") or "pending"
        reviews[status] = reviews.get(status, 0) + 1
        if (c.get("physics") or {}).get("physics_violation"):
            violations += 1
        breakdown = c.get("severity_breakdown") or {}
        layer, dist = breakdown.get("nearest_layer"), breakdown.get("nearest_layer_distance_m")
        if layer is not None and dist is not None and (nearest is None or dist < nearest[1]):
            nearest = (str(layer), float(dist), c["id"])

    top = sorted(docs, key=lambda c: -c["severity"])[:3]
    reviewed = reviews["confirmed"] + reviews["rejected"]
    class_bits = ", ".join(
        f"{cls} x{n}" for cls, n in sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    lines = [
        f"## Survey summary — {scope}",
        "",
        f"**{len(docs)} contact(s)** across {len(by_class)} class(es): {class_bits}.",
        "",
        "**Top contacts by severity:**",
        *(
            f"{i}. {c['id']} — {c['cls']}, severity {c['severity']}, "
            f"at {c['lat']:.5f}, {c['lon']:.5f}"
            for i, c in enumerate(top, start=1)
        ),
        "",
        f"**Physics violations:** {violations} of {len(docs)} contact(s) flagged.",
        f"**Review progress:** {reviews['confirmed']} confirmed, "
        f"{reviews['rejected']} rejected, {reviews['pending']} pending "
        f"({100.0 * reviewed / len(docs):.0f}% reviewed).",
    ]
    if nearest is not None:
        layer, dist, cid = nearest
        lines.append(
            f"**Sensitive layers:** closest approach is {layer.replace('_', ' ')}, "
            f"{dist:g} m from {cid}."
        )
    else:
        lines.append("**Sensitive layers:** no contact near a sensitive layer.")
    text = "\n".join(lines)

    polished = _llm_chat(
        "Polish this side-scan sonar survey summary for an operations report. "
        "Keep every number, contact id and coordinate exactly as given. "
        "Reply with markdown only.",
        text,
    )
    if polished is not None:
        return {"answer": polished, "mode": "llm"}
    return {"answer": text, "mode": "rules"}


def ask(question: str, repo: ContactRepo, layer_dir: str | Path | None = None) -> dict[str, Any]:
    """Answer a natural-language question; never raises, always answers."""
    q = question.strip().lower()

    # Summary intent first: the briefing already uses the LLM (as a prose
    # polisher only), so it must not be shadowed by LLM SQL mode.
    if _SUMMARY_RE.search(q):
        survey_match = re.search(r"survey\s+([\w.\-]+)", q)
        summary = draft_summary(repo, survey=survey_match.group(1) if survey_match else None)
        return {"sql": None, "rows": [], **summary}

    # Dimension/depth predicates are guaranteed to run in Python over the
    # contact JSON (an unmeasured height must never satisfy a size question),
    # and the LLM's SQL schema has no dims columns — so any question carrying
    # such a predicate must stay on the rule path even when an LLM is set.
    dim_filters, q = _parse_dim_filters(q)
    if not dim_filters:
        llm = _try_llm(q, repo)
        if llm is not None:
            return llm

    clauses: list[str] = []
    params: list[Any] = []
    applied: list[str] = []
    cls = _find_class(q)
    if cls:
        clauses.append("cls = ?")
        params.append(cls)
        applied.append(f"class={cls}")
    # "most severe" / "highest <anything>" are orderings, not thresholds —
    # strip superlative qualifiers entirely before scanning for band words,
    # and match the band words on word boundaries so "high" never fires
    # inside "highest".
    q_thresholds = re.sub(r"\b(?:most|least|highest|lowest)\b(?:\s+\w+)?", "", q)
    for word, threshold in _SEVERITY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", q_thresholds):
            clauses.append("severity >= ?")
            params.append(threshold)
            applied.append(f"severity>={threshold:g}")
            break
    for status in ("confirmed", "rejected", "pending"):
        if status in q:
            clauses.append("review = ?")
            params.append(status)
            applied.append(f"review={status}")
            break
    survey_match = re.search(r"survey\s+([\w.\-]+)", q)
    if survey_match:
        clauses.append("survey LIKE ?")
        params.append(f"%{survey_match.group(1)}%")
        applied.append(f"survey~{survey_match.group(1)}")
    applied.extend(f.describe() for f in dim_filters)
    note = _filters_note(applied)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    # Proximity questions run through the severity layer geometry in Python.
    # Word-boundary matching: "lane" must not fire inside "planes", nor
    # "mpa" inside "compare".
    layer_key = next(
        (v for k, v in _LAYER_WORDS.items() if re.search(rf"\b{re.escape(k)}\b", q)), None
    )
    if layer_key:
        layers = load_layers(layer_dir or REPO_ROOT / "data" / "layers")
        layer = next((la for la in layers if la.name == layer_key), None)
        rows = repo.run_sql(f"SELECT json FROM contacts{where}", tuple(params))
        docs = [row["json"] for row in rows if _passes(row["json"], dim_filters)]
        near = []
        if layer is not None:
            from geoscribe.severity import _distance_to_geometry_m

            for c in docs:
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
        return {"mode": "rules", "sql": None, "rows": near, "answer": answer + note}

    count_match = re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q)
    top_match = re.search(r"\b(?:top|first)\s*(\d+)?\b|\bhighest\b|\bmost severe\b|\bworst\b", q)

    if count_match:
        subject = cls.replace("_", " ") + " contacts" if cls else "contacts"
        if dim_filters:
            sql = f"SELECT json FROM contacts{where}"
            fetched = repo.run_sql(sql, tuple(params))
            n = sum(1 for r in fetched if _passes(r["json"], dim_filters))
            rows = [{"n": n}]
        else:
            sql = f"SELECT COUNT(*) AS n FROM contacts{where}"
            rows = repo.run_sql(sql, tuple(params))
            n = rows[0]["n"]
        return {"mode": "rules", "sql": sql, "rows": rows, "answer": f"{n} {subject} match.{note}"}

    limit = 5
    if top_match and top_match.group(1):
        limit = int(top_match.group(1))
    order = "confidence" if "confiden" in q else "severity"
    if dim_filters:
        sql = f"SELECT json FROM contacts{where} ORDER BY {order} DESC"
        fetched = repo.run_sql(sql, tuple(params))
        docs = [r["json"] for r in fetched if _passes(r["json"], dim_filters)][:limit]
        rows = [
            {
                "id": c["id"], "cls": c["cls"], "confidence": c["confidence"],
                "severity": c["severity"], "lat": c["lat"], "lon": c["lon"],
                "review": c.get("review", "pending"),
            }
            for c in docs
        ]
    else:
        sql = (
            f"SELECT id, cls, confidence, severity, lat, lon, review FROM contacts{where} "
            f"ORDER BY {order} DESC LIMIT ?"
        )
        rows = repo.run_sql(sql, (*params, limit))
    if not rows:
        return {"mode": "rules", "sql": sql, "rows": [], "answer": f"No matching contacts.{note}"}
    lines = [
        f"{r['id']}: {r['cls']} (severity {r['severity']}, confidence {r['confidence']}%)"
        for r in rows
    ]
    return {
        "mode": "rules",
        "sql": sql,
        "rows": rows,
        "answer": f"Top {len(rows)} by {order}:{note}\n" + "\n".join(lines),
    }
