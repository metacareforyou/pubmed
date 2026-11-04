"""Europe PMC REST search, used when NCBI is unavailable after retries.

It indexes the same MEDLINE records, so PMIDs from here are real PubMed IDs.
Only MEDLINE-sourced records (SRC:MED) are requested so every hit has a PMID.
"""

from __future__ import annotations

import logging

import requests

from .models import Article, Pico
from .util import Deadline, RateLimiter, get_with_retries

log = logging.getLogger("evidence")

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _group(terms: list[str]) -> str:
    quoted = [f'"{t}"' if " " in t else t for t in terms]
    return "(" + " OR ".join(quoted) + ")"


def build_query(pico: Pico, *, require_outcome: bool) -> str:
    parts = [_group(pico.population.all_terms()), _group(pico.intervention.all_terms())]
    if require_outcome and pico.outcome.present:
        parts.append(_group(pico.outcome.all_terms()))
    return " AND ".join(parts) + " AND SRC:MED AND HAS_ABSTRACT:Y"


class EuropePmcClient:
    def __init__(self, session: requests.Session | None = None, sleep=None):
        self.session = session or requests.Session()
        self.limiter = RateLimiter(5.0)
        self._sleep = sleep

    def search(self, query: str, *, page_size: int, deadline: Deadline) -> list[Article]:
        kwargs = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        response = get_with_retries(
            self.session,
            SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": str(page_size),
                "sort": "RELEVANCE",
            },
            service="europepmc",
            deadline=deadline,
            limiter=self.limiter,
            **kwargs,
        )
        return parse_results(response.json())


def parse_results(payload: dict) -> list[Article]:
    out: list[Article] = []
    for item in payload.get("resultList", {}).get("result", []):
        pmid = str(item.get("pmid") or "").strip()
        if not pmid:
            continue
        year = None
        try:
            year = int(item.get("pubYear"))
        except (TypeError, ValueError):
            pass
        pub_types = [
            t for t in (item.get("pubTypeList") or {}).get("pubType", []) if isinstance(t, str)
        ]
        mesh = [
            h.get("descriptorName", "")
            for h in (item.get("meshHeadingList") or {}).get("meshHeading", [])
            if isinstance(h, dict)
        ]
        keywords = [
            k for k in (item.get("keywordList") or {}).get("keyword", []) if isinstance(k, str)
        ]
        authors = [a.strip() for a in (item.get("authorString") or "").rstrip(".").split(",") if a.strip()]
        out.append(
            Article(
                pmid=pmid,
                title=(item.get("title") or "").rstrip("."),
                abstract=item.get("abstractText") or "",
                journal=item.get("journalTitle") or "",
                authors=authors,
                year=year,
                date=str(year) if year else "",
                publication_types=pub_types,
                mesh_terms=[m for m in mesh if m],
                keywords=keywords,
                doi=item.get("doi") or "",
                pmcid=item.get("pmcid") or "",
                source="europepmc",
            )
        )
    return out
