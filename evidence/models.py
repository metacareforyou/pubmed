"""Plain data carried between the stages."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    date: str = ""
    publication_types: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    doi: str = ""
    pmcid: str = ""
    source: str = "pubmed"
    # Filled in by ranking.
    design: str = "other"
    design_score: float = 0.0
    relevance_score: float = 0.0
    recency_score: float = 0.0
    score: float = 0.0

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["url"] = self.url
        return data

    def to_legacy_tuple(self) -> list[Any]:
        """The array shape the Metacare.ai PicoForm page renders:
        [title, abstract, journal, authors[], date, keywords, mesh_terms]."""
        return [
            self.title,
            self.abstract,
            self.journal,
            list(self.authors),
            self.date or (str(self.year) if self.year else ""),
            ", ".join(self.keywords),
            ", ".join(self.mesh_terms),
        ]


@dataclass
class PicoElement:
    text: str
    synonyms: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return bool(self.text.strip())

    def all_terms(self) -> list[str]:
        seen: list[str] = []
        for term in [self.text, *self.synonyms]:
            term = term.strip()
            if term and term.lower() not in {s.lower() for s in seen}:
                seen.append(term)
        return seen


@dataclass
class Pico:
    population: PicoElement
    intervention: PicoElement
    comparison: PicoElement
    outcome: PicoElement
    question_type: str = "therapy"  # therapy | diagnosis | prognosis | etiology
    rewritten_question: str = ""

    def as_text(self) -> str:
        parts = [
            f"P: {self.population.text}",
            f"I: {self.intervention.text}",
            f"C: {self.comparison.text or 'not specified'}",
            f"O: {self.outcome.text}",
        ]
        return "\n".join(parts)

    def search_text(self) -> str:
        return " ".join(
            term
            for element in (self.population, self.intervention, self.comparison, self.outcome)
            for term in element.all_terms()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": asdict(self.population),
            "intervention": asdict(self.intervention),
            "comparison": asdict(self.comparison),
            "outcome": asdict(self.outcome),
            "question_type": self.question_type,
            "rewritten_question": self.rewritten_question,
        }


@dataclass
class Quote:
    pmid: str
    text: str


@dataclass
class Finding:
    statement: str
    pmids: list[str]
    quotes: list[Quote] = field(default_factory=list)
    verified: bool = False
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "pmids": list(self.pmids),
            "quotes": [asdict(q) for q in self.quotes],
            "verified": self.verified,
        }


@dataclass
class Synthesis:
    bottom_line: str
    findings: list[Finding]
    certainty: str  # high | moderate | low | very_low | insufficient
    certainty_rationale: str
    limitations: list[str] = field(default_factory=list)
    insufficient: bool = False
    stated_certainty: str = ""
    certainty_cap: str = ""
    cap_reason: str = ""
