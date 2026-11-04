"""PubMed query ladder.

Synonyms are ORed within each PICO element. Population and intervention are
required; comparison and outcome are optional and dropped first. Each step
starts with the systematic review subset, then PubMed Clinical Queries
filters, then no filter, relaxing until there are enough hits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .models import Article, Pico
from .util import Deadline, UpstreamError

log = logging.getLogger("evidence")

SYSTEMATIC = "systematic[sb]"

# PubMed Clinical Queries filters (Haynes et al.), by question type.
CLINICAL_QUERY = {
    "therapy": {
        "narrow": "(randomized controlled trial[Publication Type] OR (randomized[Title/Abstract] AND controlled[Title/Abstract] AND trial[Title/Abstract]))",
        "broad": "((clinical[Title/Abstract] AND trial[Title/Abstract]) OR clinical trials as topic[MeSH Terms] OR clinical trial[Publication Type] OR random*[Title/Abstract] OR random allocation[MeSH Terms] OR therapeutic use[MeSH Subheading])",
    },
    "diagnosis": {
        "narrow": "(specificity[Title/Abstract])",
        "broad": "(sensitiv*[Title/Abstract] OR sensitivity and specificity[MeSH Terms] OR diagnos*[Title/Abstract] OR diagnosis[MeSH:noexp] OR diagnostic *[MeSH:noexp] OR diagnosis,differential[MeSH:noexp] OR diagnosis[Subheading:noexp])",
    },
    "prognosis": {
        "narrow": "(prognos*[Title/Abstract] OR (first[Title/Abstract] AND episode[Title/Abstract]) OR cohort[Title/Abstract])",
        "broad": "(incidence[MeSH:noexp] OR mortality[MeSH Terms] OR follow up studies[MeSH:noexp] OR prognos*[Text Word] OR predict*[Text Word] OR course*[Text Word])",
    },
    "etiology": {
        "narrow": "((relative[Title/Abstract] AND risk*[Title/Abstract]) OR (relative risk[Text Word]) OR risks[Text Word] OR cohort studies[MeSH:noexp] OR (cohort[Title/Abstract] AND stud*[Title/Abstract]))",
        "broad": "(risk*[Title/Abstract] OR risk*[MeSH:noexp] OR risk *[MeSH:noexp] OR cohort studies[MeSH Terms] OR group*[Text Word])",
    },
}

HUMAN_ENGLISH = "humans[MeSH Terms] AND english[Language] AND hasabstract[text]"


def _quote(term: str) -> str:
    term = term.replace('"', "").strip()
    return f'"{term}"' if " " in term else term


def element_clause(terms: list[str], mesh: list[str]) -> str:
    """OR of synonyms in title/abstract plus any MeSH headings. Empty -> ''."""
    pieces = [f"{_quote(t)}[tiab]" for t in terms]
    pieces += [f"{_quote(m)}[mh]" for m in mesh if m.strip()]
    if not pieces:
        return ""
    return "(" + " OR ".join(pieces) + ")"


@dataclass
class LadderStep:
    name: str
    query: str
    hits: int = 0
    pmids: list[str] = field(default_factory=list)


def build_steps(pico: Pico) -> list[LadderStep]:
    p = element_clause(pico.population.all_terms(), pico.population.mesh)
    i = element_clause(pico.intervention.all_terms(), pico.intervention.mesh)
    c = element_clause(pico.comparison.all_terms(), pico.comparison.mesh)
    o = element_clause(pico.outcome.all_terms(), pico.outcome.mesh)
    if not p or not i:
        raise ValueError("population and intervention are required to search")

    cq = CLINICAL_QUERY.get(pico.question_type, CLINICAL_QUERY["therapy"])
    core_variants = []
    if c and o:
        core_variants.append(("PICO", f"{p} AND {i} AND {c} AND {o}"))
    if o:
        core_variants.append(("PIO", f"{p} AND {i} AND {o}"))
    core_variants.append(("PI", f"{p} AND {i}"))

    steps: list[LadderStep] = []
    for label, core in core_variants:
        steps.append(LadderStep(f"{label} systematic reviews", f"{core} AND {SYSTEMATIC} AND {HUMAN_ENGLISH}"))
        steps.append(LadderStep(f"{label} clinical queries narrow", f"{core} AND {cq['narrow']} AND {HUMAN_ENGLISH}"))
    for label, core in core_variants:
        steps.append(LadderStep(f"{label} clinical queries broad", f"{core} AND {cq['broad']} AND {HUMAN_ENGLISH}"))
    steps.append(LadderStep("PI any design", f"{p} AND {i} AND {HUMAN_ENGLISH}"))
    steps.append(LadderStep("PI any design, any language", f"{p} AND {i}"))
    return steps


@dataclass
class RetrievalResult:
    articles: list[Article]
    steps_run: list[LadderStep]
    selected_step: str
    source: str  # pubmed | europepmc
    fallback_reason: str = ""


def retrieve(
    pico: Pico,
    *,
    pubmed,
    europepmc,
    deadline: Deadline,
    min_hits: int,
    fetch_max: int,
) -> RetrievalResult:
    """Walk the ladder on PubMed; fall back to Europe PMC if NCBI fails."""
    steps = build_steps(pico)
    run: list[LadderStep] = []
    pool: dict[str, None] = {}
    selected = ""
    try:
        for step in steps:
            deadline.check("pubmed ladder")
            step.hits, step.pmids = pubmed.search(step.query, retmax=fetch_max, deadline=deadline)
            run.append(step)
            for pmid in step.pmids:
                pool.setdefault(pmid, None)
            if not selected and step.hits >= min_hits:
                selected = step.name
            exhausted = step.hits <= len(step.pmids)  # this step returned everything it had
            if len(pool) >= fetch_max or (selected and (len(pool) >= min_hits or exhausted)):
                break
        if not selected and run:
            selected = max(run, key=lambda s: s.hits).name
        pmids = list(pool)[:fetch_max]
        articles = pubmed.fetch(pmids, deadline=deadline) if pmids else []
        articles = [a for a in articles if a.abstract.strip()]
        return RetrievalResult(articles, run, selected or "none", "pubmed")
    except UpstreamError as exc:
        log.warning("pubmed unavailable (%s); trying Europe PMC", exc.detail)
        reason = exc.detail

    from .europepmc import build_query  # local import keeps this module light

    for require_outcome in (True, False):
        deadline.check("europepmc search")
        query = build_query(pico, require_outcome=require_outcome)
        articles = europepmc.search(query, page_size=fetch_max, deadline=deadline)
        articles = [a for a in articles if a.abstract.strip()]
        run.append(LadderStep(f"europepmc {'PIO' if require_outcome else 'PI'}", query, len(articles), [a.pmid for a in articles]))
        if len(articles) >= min_hits or not require_outcome:
            return RetrievalResult(articles, run, run[-1].name, "europepmc", fallback_reason=reason)
    return RetrievalResult([], run, "none", "europepmc", fallback_reason=reason)
