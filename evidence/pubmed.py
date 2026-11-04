"""NCBI E-utilities client: esearch + efetch, with rate limiting and retries.

Without an API key NCBI allows 3 requests a second, with one 10. A free key
comes from an NCBI account; see https://www.ncbi.nlm.nih.gov/books/NBK25497/.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Iterable

import requests

from .models import Article
from .util import Deadline, RateLimiter, get_with_retries

log = logging.getLogger("evidence")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "metacare-evidence"


class PubMedClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        email: str,
        session: requests.Session | None = None,
        sleep=None,
    ):
        self.api_key = api_key
        self.email = email
        self.session = session or requests.Session()
        self.limiter = RateLimiter(9.0 if api_key else 2.5)
        self._sleep = sleep

    def _common(self) -> dict[str, str]:
        params = {"tool": TOOL_NAME, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, path: str, params: dict, deadline: Deadline) -> requests.Response:
        kwargs = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        return get_with_retries(
            self.session,
            f"{EUTILS}/{path}",
            params={**self._common(), **params},
            service="pubmed",
            deadline=deadline,
            limiter=self.limiter,
            **kwargs,
        )

    def search(self, query: str, *, retmax: int, deadline: Deadline) -> tuple[int, list[str]]:
        """Returns (total count, pmids) for a PubMed query, relevance sorted."""
        response = self._get(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": str(retmax),
                "retmode": "json",
                "sort": "relevance",
            },
            deadline,
        )
        data = response.json().get("esearchresult", {})
        count = int(data.get("count", "0") or 0)
        ids = [str(i) for i in data.get("idlist", [])]
        return count, ids

    def fetch(self, pmids: Iterable[str], *, deadline: Deadline) -> list[Article]:
        pmids = [p for p in pmids if p]
        if not pmids:
            return []
        response = self._get(
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
            deadline,
        )
        return parse_efetch_xml(response.text)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _abstract_text(article: ET.Element) -> str:
    parts = []
    for section in article.findall("./Abstract/AbstractText"):
        label = section.get("Label")
        body = _text(section)
        if not body:
            continue
        parts.append(f"{label.title()}: {body}" if label and label.upper() != "UNLABELLED" else body)
    return "\n".join(parts)


def _year(article: ET.Element) -> tuple[int | None, str]:
    pubdate = article.find("./Journal/JournalIssue/PubDate")
    if pubdate is None:
        return None, ""
    year = _text(pubdate.find("Year"))
    month = _text(pubdate.find("Month"))
    medline = _text(pubdate.find("MedlineDate"))
    if not year and medline:
        match = re.search(r"\b(19|20)\d{2}\b", medline)
        year = match.group(0) if match else ""
    date = " ".join(p for p in [year, month] if p)
    try:
        return int(year), date
    except ValueError:
        return None, date


def parse_efetch_xml(xml_text: str) -> list[Article]:
    root = ET.fromstring(xml_text)
    out: list[Article] = []
    for item in root.findall("./PubmedArticle"):
        citation = item.find("MedlineCitation")
        if citation is None:
            continue
        article = citation.find("Article")
        if article is None:
            continue
        pmid = _text(citation.find("PMID"))
        year, date = _year(article)
        authors = []
        for author in article.findall("./AuthorList/Author"):
            last = _text(author.find("LastName"))
            initials = _text(author.find("Initials"))
            collective = _text(author.find("CollectiveName"))
            name = " ".join(p for p in [last, initials] if p) or collective
            if name:
                authors.append(name)
        ids = {
            (node.get("IdType") or "").lower(): _text(node)
            for node in item.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        out.append(
            Article(
                pmid=pmid,
                title=_text(article.find("ArticleTitle")).rstrip("."),
                abstract=_abstract_text(article),
                journal=_text(article.find("./Journal/ISOAbbreviation"))
                or _text(article.find("./Journal/Title")),
                authors=authors,
                year=year,
                date=date,
                publication_types=[
                    _text(node) for node in article.findall("./PublicationTypeList/PublicationType")
                ],
                mesh_terms=[
                    _text(node.find("DescriptorName"))
                    for node in citation.findall("./MeshHeadingList/MeshHeading")
                ],
                keywords=[
                    _text(node) for node in citation.findall("./KeywordList/Keyword") if _text(node)
                ],
                doi=ids.get("doi", ""),
                pmcid=ids.get("pmc", ""),
                source="pubmed",
            )
        )
    return out
