"""Turn a verified synthesis into the Markdown summary and the API payloads."""

from __future__ import annotations

from typing import Any

from .models import Article, Pico, Synthesis

CERTAINTY_LABEL = {
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
    "very_low": "Very low",
    "insufficient": "Insufficient",
}

DISCLAIMER = (
    "This report summarizes published abstracts for education and information. "
    "It is not medical advice and does not replace clinical judgment."
)


def _cite(pmids: list[str]) -> str:
    return " ".join(f"[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in pmids)


def render_markdown(pico: Pico, synthesis: Synthesis, articles: list[Article], *, source: str) -> str:
    lines: list[str] = []
    lines.append("## Bottom line")
    if synthesis.insufficient and not any(f.verified for f in synthesis.findings):
        lines.append(
            "The retrieved abstracts do not answer this question well enough to state a finding. "
            + (synthesis.certainty_rationale or "")
        )
    else:
        lines.append(synthesis.bottom_line)
    lines.append("")
    lines.append(f"**Certainty of evidence: {CERTAINTY_LABEL.get(synthesis.certainty, synthesis.certainty)}**")
    if synthesis.certainty_cap and synthesis.certainty_cap not in ("high", "insufficient") and synthesis.cap_reason:
        lines.append(f"Capped at {CERTAINTY_LABEL[synthesis.certainty_cap].lower()} because the strongest supporting design is {synthesis.cap_reason}.")
    if synthesis.certainty_rationale and not synthesis.insufficient:
        lines.append(synthesis.certainty_rationale)
    lines.append("")

    verified = [f for f in synthesis.findings if f.verified]
    if verified:
        lines.append("## Verified findings")
        for finding in verified:
            lines.append(f"- {finding.statement} {_cite(finding.pmids)}")
            for quote in finding.quotes[:2]:
                lines.append(f'  - "{quote.text}" (PMID {quote.pmid})')
        lines.append("")
    dropped = [f for f in synthesis.findings if not f.verified]
    if dropped:
        lines.append(f"_{len(dropped)} draft finding{'s' if len(dropped) != 1 else ''} removed because the abstracts did not support {'them' if len(dropped) != 1 else 'it'} word for word._")
        lines.append("")

    if synthesis.limitations:
        lines.append("## Limitations")
        for item in synthesis.limitations:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## PICO")
    lines.append(f"- **Population:** {pico.population.text}")
    lines.append(f"- **Intervention:** {pico.intervention.text}")
    lines.append(f"- **Comparison:** {pico.comparison.text or 'not specified'}")
    lines.append(f"- **Outcome:** {pico.outcome.text}")
    lines.append("")

    lines.append("## Studies considered")
    for a in articles:
        who = (a.authors[0] + (" et al." if len(a.authors) > 1 else "")) if a.authors else a.journal
        lines.append(f"- {who} {a.year or ''}. {a.title}. _{a.journal}_. {a.design.replace('_', ' ')}. [PMID {a.pmid}]({a.url})")
    lines.append("")
    lines.append(f"_Source: {'PubMed' if source == 'pubmed' else 'Europe PMC (PubMed was unavailable)'}. {DISCLAIMER}_")
    return "\n".join(lines)


def insufficient_markdown(pico: Pico | None, reason: str) -> str:
    lines = ["## Bottom line", reason, "", "**Certainty of evidence: Insufficient**", ""]
    if pico is not None:
        lines += [
            "## PICO",
            f"- **Population:** {pico.population.text}",
            f"- **Intervention:** {pico.intervention.text}",
            f"- **Comparison:** {pico.comparison.text or 'not specified'}",
            f"- **Outcome:** {pico.outcome.text}",
            "",
        ]
    lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines)


def evidence_payload(*, pico: Pico | None, synthesis: Synthesis | None, articles: list[Article],
                     summary: str, source: str, ladder: list[dict[str, Any]], request_id: str,
                     timings: dict[str, float], status: str) -> dict[str, Any]:
    verified = [f for f in (synthesis.findings if synthesis else []) if f.verified]
    return {
        "status": status,  # ok | insufficient
        "request_id": request_id,
        "pico": pico.to_dict() if pico else None,
        "pico_question": pico.as_text() if pico else "",
        "bottom_line": (synthesis.bottom_line if synthesis and verified else ""),
        "certainty": synthesis.certainty if synthesis else "insufficient",
        "certainty_cap": synthesis.certainty_cap if synthesis else "",
        "cap_reason": synthesis.cap_reason if synthesis else "",
        "findings": [f.to_dict() for f in verified],
        "dropped_findings": len([f for f in (synthesis.findings if synthesis else []) if not f.verified]),
        "limitations": synthesis.limitations if synthesis else [],
        "articles": [a.to_dict() for a in articles],
        "summary": summary,
        "retrieval": {"source": source, "ladder": ladder},
        "timings_ms": {k: round(v * 1000) for k, v in timings.items()},
        "disclaimer": DISCLAIMER,
    }
