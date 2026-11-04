"""The two Claude calls: PICO extraction and evidence synthesis.

Both use structured outputs (output_config.format) so the response is valid
JSON matching a schema, and server-side refusal fallbacks so a safeguard
decline is re-run on a fallback model inside the same call. A final refusal
is returned as `None` and reported as insufficient evidence, never as a 500.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from .models import Article, Finding, Pico, PicoElement, Quote, Synthesis
from .util import Deadline

log = logging.getLogger("evidence")

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LlmUnavailable(Exception):
    """Claude could not be called (auth, quota, outage). Message is safe to log."""


def make_client(api_key: str | None, timeout: float = 60.0) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)


# ---------------------------------------------------------------------------
# PICO
# ---------------------------------------------------------------------------

_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The element in plain clinical words; empty string if the question does not state one."},
        "synonyms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 6 alternative search terms: common names, drug generic and class names, abbreviations spelled out, British and American spellings. Short phrases suitable for a PubMed title/abstract search.",
        },
        "mesh": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 3 exact MeSH descriptor names if you are confident they exist, else empty.",
        },
    },
    "required": ["text", "synonyms", "mesh"],
    "additionalProperties": False,
}

PICO_SCHEMA = {
    "type": "object",
    "properties": {
        "population": _ELEMENT_SCHEMA,
        "intervention": _ELEMENT_SCHEMA,
        "comparison": _ELEMENT_SCHEMA,
        "outcome": _ELEMENT_SCHEMA,
        "question_type": {"type": "string", "enum": ["therapy", "diagnosis", "prognosis", "etiology"]},
        "rewritten_question": {"type": "string", "description": "One sentence, the question restated as an answerable PICO question."},
        "answerable": {"type": "boolean", "description": "False if this is not a clinical question the medical literature could answer."},
    },
    "required": ["population", "intervention", "comparison", "outcome", "question_type", "rewritten_question", "answerable"],
    "additionalProperties": False,
}

PICO_SYSTEM = (
    "You turn clinical questions into PICO search frames for a PubMed search. "
    "Population is the patient group or condition. Intervention is the exposure, test, drug or procedure asked about. "
    "Comparison is the alternative, if the question names one; otherwise leave it empty rather than inventing one. "
    "Outcome is what is being measured; if the question gives none, choose the outcome a clinician would most want, "
    "and say so in rewritten_question. Keep each element short. Do not include the population inside the intervention. "
    "Synonyms must be real terms used in titles and abstracts, not sentences. Never include personal identifiers that appear in the question."
)


def extract_pico(client: anthropic.Anthropic, model: str, question: str, deadline: Deadline) -> Pico | None:
    """Returns None when the question is not answerable, or when Claude refused."""
    data = _structured_call(
        client,
        model=model,
        system=PICO_SYSTEM,
        user=f"Clinical question:\n{question}",
        schema=PICO_SCHEMA,
        max_tokens=2000,
        effort="low",
        timeout=deadline.timeout(45.0),
    )
    if data is None or not data.get("answerable", True):
        return None

    def element(key: str) -> PicoElement:
        raw = data.get(key) or {}
        return PicoElement(
            text=str(raw.get("text", "")).strip(),
            synonyms=[str(s).strip() for s in raw.get("synonyms", []) if str(s).strip()][:6],
            mesh=[str(m).strip() for m in raw.get("mesh", []) if str(m).strip()][:3],
        )

    pico = Pico(
        population=element("population"),
        intervention=element("intervention"),
        comparison=element("comparison"),
        outcome=element("outcome"),
        question_type=data.get("question_type", "therapy"),
        rewritten_question=str(data.get("rewritten_question", "")).strip(),
    )
    if not pico.population.present or not pico.intervention.present:
        return None
    return pico


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "bottom_line": {"type": "string", "description": "Two to four sentences answering the PICO question from the abstracts only. Markdown allowed. No citations here; they go in findings."},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "One finding in one or two sentences. Any number you state must appear verbatim in a cited abstract."},
                    "pmids": {"type": "array", "items": {"type": "string"}, "description": "PMIDs of the abstracts that support this statement, from the provided list only."},
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pmid": {"type": "string"},
                                "text": {"type": "string", "description": "An exact, contiguous excerpt (5 to 40 words) copied from that abstract or title that supports the statement."},
                            },
                            "required": ["pmid", "text"],
                            "additionalProperties": False,
                        },
                        "description": "At least one exact quote per cited PMID.",
                    },
                },
                "required": ["statement", "pmids", "quotes"],
                "additionalProperties": False,
            },
        },
        "certainty": {"type": "string", "enum": ["high", "moderate", "low", "very_low", "insufficient"], "description": "GRADE-style certainty in the bottom line, judged from study designs, consistency, directness and precision of what the abstracts report."},
        "certainty_rationale": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "insufficient": {"type": "boolean", "description": "True if the abstracts do not address the question well enough to give a bottom line."},
    },
    "required": ["bottom_line", "findings", "certainty", "certainty_rationale", "limitations", "insufficient"],
    "additionalProperties": False,
}

SYNTHESIS_SYSTEM = (
    "You are an evidence-based medicine methodologist writing for clinicians. "
    "Use only the numbered abstracts provided. Do not use outside knowledge, do not cite anything not in the list, "
    "and do not invent effect sizes, confidence intervals, sample sizes or p values: every number in a statement must be "
    "copied from a cited abstract. Each finding must carry at least one exact quote from each PMID it cites; a quote is a "
    "contiguous excerpt copied character for character. Prefer systematic reviews and randomized trials; say when evidence "
    "is observational. If the abstracts do not answer the question, set insufficient to true and say what is missing. "
    "Write plainly. Do not give treatment recommendations for a specific patient."
)


def _abstracts_block(articles: list[Article]) -> str:
    parts = []
    for n, a in enumerate(articles, 1):
        parts.append(
            f"[{n}] PMID {a.pmid} | {a.year or 'n.d.'} | {a.journal} | design: {a.design}\n"
            f"Title: {a.title}\nAbstract: {a.abstract}\n"
        )
    return "\n".join(parts)


def synthesize(client: anthropic.Anthropic, model: str, pico: Pico, articles: list[Article], deadline: Deadline) -> Synthesis | None:
    user = (
        f"PICO question:\n{pico.as_text()}\n\n"
        f"Restated: {pico.rewritten_question}\n\n"
        f"Abstracts ({len(articles)}):\n\n{_abstracts_block(articles)}"
    )
    data = _structured_call(
        client,
        model=model,
        system=SYNTHESIS_SYSTEM,
        user=user,
        schema=SYNTHESIS_SCHEMA,
        max_tokens=6000,
        effort="medium",
        timeout=deadline.timeout(75.0),
    )
    if data is None:
        return None
    findings = []
    for raw in data.get("findings", []):
        findings.append(
            Finding(
                statement=str(raw.get("statement", "")).strip(),
                pmids=[str(p).strip() for p in raw.get("pmids", []) if str(p).strip()],
                quotes=[Quote(pmid=str(q.get("pmid", "")).strip(), text=str(q.get("text", "")).strip()) for q in raw.get("quotes", [])],
            )
        )
    return Synthesis(
        bottom_line=str(data.get("bottom_line", "")).strip(),
        findings=findings,
        certainty=data.get("certainty", "insufficient"),
        certainty_rationale=str(data.get("certainty_rationale", "")).strip(),
        limitations=[str(l).strip() for l in data.get("limitations", []) if str(l).strip()],
        insufficient=bool(data.get("insufficient", False)),
        stated_certainty=data.get("certainty", "insufficient"),
    )


# ---------------------------------------------------------------------------
# Shared call
# ---------------------------------------------------------------------------

def _structured_call(client: anthropic.Anthropic, *, model: str, system: str, user: str, schema: dict,
                     max_tokens: int, effort: str, timeout: float) -> dict[str, Any] | None:
    try:
        response = client.with_options(timeout=timeout).beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as exc:
        raise LlmUnavailable("claude authentication failed") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LlmUnavailable("claude key lacks permission") from exc
    except anthropic.RateLimitError as exc:
        raise LlmUnavailable("claude rate limited") from exc
    except anthropic.APITimeoutError as exc:
        raise LlmUnavailable("claude timed out") from exc
    except anthropic.APIStatusError as exc:
        raise LlmUnavailable(f"claude returned {exc.status_code}") from exc
    except anthropic.APIConnectionError as exc:
        raise LlmUnavailable("claude unreachable") from exc

    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        log.warning("claude refused (category=%s)", category)
        return None
    if response.stop_reason == "max_tokens":
        log.warning("claude output truncated at max_tokens")
        return None
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("claude returned non-JSON despite output_config")
        return None
