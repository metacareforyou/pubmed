"""Verification of the synthesis against the retrieved abstracts.

A finding survives only if every PMID it cites was retrieved, every quote is
found in the cited record, and every number in the statement appears in at
least one cited abstract. Certainty is then capped by the weakest design
among the surviving citations, whatever the model claimed.
"""

from __future__ import annotations

import re
import unicodedata

from .models import Article, Finding, Synthesis
from .rank import DESIGN_CERTAINTY_CAP

CERTAINTY_ORDER = ["very_low", "low", "moderate", "high"]

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?!\w)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[“”\"'`´‘’]", "", text)
    text = re.sub(r"[^a-z0-9%.<>=+\-/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, record: str) -> bool:
    q = normalize(quote)
    if len(q) < 8:
        return False
    r = normalize(record)
    if q in r:
        return True
    # Tolerate a trimmed or lightly edited quote: all 4-grams of the quote present in order of 80%.
    words = q.split()
    if len(words) < 6:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    hit = sum(1 for g in grams if g in r)
    return hit / len(grams) >= 0.8


def numbers_in(statement: str) -> list[str]:
    out = []
    for raw in _NUMBER.findall(statement):
        value = raw.replace(",", ".")
        if value.endswith("."):
            value = value[:-1]
        out.append(value)
    return out


def _record_text(article: Article) -> str:
    return f"{article.title}\n{article.abstract}"


def _number_present(number: str, text: str) -> bool:
    plain = normalize(text).replace(",", ".")
    pattern = r"(?<![\d.])" + re.escape(number) + r"(?![\d])"
    if re.search(pattern, plain):
        return True
    # 0.45 vs .45, 23 vs 23.0
    try:
        f = float(number)
    except ValueError:
        return False
    alternates = {f"{f:g}", f"{f:.1f}", f"{f:.2f}", str(int(f)) if f == int(f) else ""}
    return any(a and re.search(r"(?<![\d.])" + re.escape(a) + r"(?![\d])", plain) for a in alternates)


def verify_finding(finding: Finding, by_pmid: dict[str, Article]) -> Finding:
    if not finding.pmids:
        finding.rejected_reason = "no citation"
        return finding
    unknown = [p for p in finding.pmids if p not in by_pmid]
    if unknown:
        finding.rejected_reason = f"cites PMIDs not retrieved: {', '.join(unknown)}"
        return finding
    quoted_pmids = set()
    for quote in finding.quotes:
        article = by_pmid.get(quote.pmid)
        if article is None or quote.pmid not in finding.pmids:
            finding.rejected_reason = f"quote attributed to uncited PMID {quote.pmid}"
            return finding
        if not quote_in(quote.text, _record_text(article)):
            finding.rejected_reason = f"quote not found in PMID {quote.pmid}"
            return finding
        quoted_pmids.add(quote.pmid)
    missing = [p for p in finding.pmids if p not in quoted_pmids]
    if missing:
        finding.rejected_reason = f"no quote for PMID {', '.join(missing)}"
        return finding
    cited_text = "\n".join(_record_text(by_pmid[p]) for p in finding.pmids)
    for number in numbers_in(finding.statement):
        if not _number_present(number, cited_text):
            finding.rejected_reason = f"number {number} not in cited abstracts"
            return finding
    finding.verified = True
    return finding


def weakest_cap(articles: list[Article]) -> tuple[str, str]:
    """Lowest certainty ceiling among the given designs, and which design set it."""
    cap, reason = "high", ""
    for article in articles:
        ceiling = DESIGN_CERTAINTY_CAP.get(article.design, "low")
        if CERTAINTY_ORDER.index(ceiling) < CERTAINTY_ORDER.index(cap):
            cap, reason = ceiling, f"{article.design.replace('_', ' ')} (PMID {article.pmid})"
    return cap, reason


def verify(synthesis: Synthesis, articles: list[Article]) -> Synthesis:
    by_pmid = {a.pmid: a for a in articles}
    for finding in synthesis.findings:
        verify_finding(finding, by_pmid)
    surviving = [f for f in synthesis.findings if f.verified]
    if not surviving:
        synthesis.insufficient = True
        synthesis.certainty = "insufficient"
        synthesis.certainty_cap = "insufficient"
        synthesis.cap_reason = "no finding survived verification"
        return synthesis
    cited = {p for f in surviving for p in f.pmids}
    cap, reason = weakest_cap([by_pmid[p] for p in cited])
    synthesis.certainty_cap, synthesis.cap_reason = cap, reason
    stated = synthesis.stated_certainty or synthesis.certainty
    if stated in CERTAINTY_ORDER and CERTAINTY_ORDER.index(stated) > CERTAINTY_ORDER.index(cap):
        synthesis.certainty = cap
    elif stated in CERTAINTY_ORDER:
        synthesis.certainty = stated
    else:
        synthesis.certainty = cap if not synthesis.insufficient else "insufficient"
    return synthesis
