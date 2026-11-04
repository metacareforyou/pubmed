"""Ranking: BM25 relevance + study design + recency. Transparent and dependency free."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date

from .models import Article

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = set(
    "a an and are as at be by for from has in is it its of on or that the to was were with "
    "which this these those than then there their we our study studies patients patient "
    "results methods conclusions background objective aim aims compared versus vs".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], index: int) -> float:
        tf = self.tf[index]
        dl = len(self.docs[index])
        total = 0.0
        for term in query:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return total


# Study design, from PubMed publication types (and MeSH where types are thin).
# Ordered strongest first; the first match wins.
DESIGNS: list[tuple[str, float, tuple[str, ...]]] = [
    ("systematic_review", 1.00, ("meta-analysis", "systematic review", "network meta-analysis")),
    ("guideline", 0.90, ("practice guideline", "guideline", "consensus development conference")),
    ("rct", 0.85, ("randomized controlled trial", "pragmatic clinical trial", "equivalence trial")),
    ("clinical_trial", 0.65, ("clinical trial", "controlled clinical trial", "clinical trial, phase iii", "clinical trial, phase ii", "adaptive clinical trial")),
    ("cohort", 0.50, ("cohort studies", "observational study", "prospective studies", "longitudinal studies", "follow-up studies", "comparative study")),
    ("case_control", 0.40, ("case-control studies", "cross-sectional studies", "retrospective studies")),
    ("narrative_review", 0.30, ("review", "editorial", "comment", "letter")),
    ("case_report", 0.10, ("case reports",)),
]

# GRADE starting points and the ceiling each design can support.
DESIGN_CERTAINTY_CAP = {
    "systematic_review": "high",
    "guideline": "high",
    "rct": "high",
    "clinical_trial": "moderate",
    "cohort": "low",
    "case_control": "low",
    "other": "low",
    "narrative_review": "very_low",
    "case_report": "very_low",
}


def classify_design(article: Article) -> tuple[str, float]:
    haystack = {t.lower() for t in article.publication_types} | {m.lower() for m in article.mesh_terms}
    title = article.title.lower()
    for name, weight, needles in DESIGNS:
        if any(n in haystack for n in needles):
            return name, weight
    # Titles often say what publication types miss.
    if "meta-analysis" in title or "systematic review" in title:
        return "systematic_review", 1.0
    if "randomized" in title or "randomised" in title:
        return "rct", 0.85
    if "cohort" in title or "registry" in title:
        return "cohort", 0.5
    if "case report" in title:
        return "case_report", 0.1
    return "other", 0.35


def recency(year: int | None, today: date | None = None, half_life_years: float = 8.0) -> float:
    if not year:
        return 0.3
    age = max(0, (today or date.today()).year - year)
    return 0.5 ** (age / half_life_years)


def rank(articles: list[Article], query_text: str, *, top_k: int, today: date | None = None,
         w_relevance: float = 0.5, w_design: float = 0.3, w_recency: float = 0.2) -> list[Article]:
    if not articles:
        return []
    docs = [tokenize(f"{a.title} {a.title} {a.abstract} {' '.join(a.mesh_terms)}") for a in articles]
    bm25 = BM25(docs)
    query = tokenize(query_text)
    raw = [bm25.score(query, i) for i in range(len(articles))]
    top = max(raw) or 1.0
    for article, score in zip(articles, raw):
        article.relevance_score = score / top
        article.design, article.design_score = classify_design(article)
        article.recency_score = recency(article.year, today)
        article.score = (
            w_relevance * article.relevance_score
            + w_design * article.design_score
            + w_recency * article.recency_score
        )
    ordered = sorted(articles, key=lambda a: a.score, reverse=True)
    return ordered[:top_k]
