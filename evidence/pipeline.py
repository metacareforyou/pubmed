"""One request end to end: PICO -> ladder -> rank -> synthesis -> verify -> report."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import llm, report
from .cache import TtlCache, question_key
from .config import Settings
from .europepmc import EuropePmcClient
from .ladder import retrieve
from .models import Article, Pico, Synthesis
from .pubmed import PubMedClient
from .rank import rank
from .util import Deadline, DeadlineExceeded, UpstreamError
from .verify import verify

log = logging.getLogger("evidence")


@dataclass
class EvidenceResult:
    status: str  # ok | insufficient
    pico: Pico | None
    synthesis: Synthesis | None
    articles: list[Article]
    summary: str
    source: str
    ladder: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    request_id: str = ""

    def payload(self) -> dict[str, Any]:
        return report.evidence_payload(
            pico=self.pico, synthesis=self.synthesis, articles=self.articles, summary=self.summary,
            source=self.source, ladder=self.ladder, request_id=self.request_id, timings=self.timings,
            status=self.status,
        )

    def legacy_payload(self) -> dict[str, Any]:
        """What the Metacare.ai PicoForm page has always consumed."""
        return {
            "pico_question": self.pico.as_text() if self.pico else "",
            "articles": [a.to_legacy_tuple() for a in self.articles],
            "summary": self.summary,
        }


class EvidenceService:
    def __init__(self, settings: Settings, *, client=None, pubmed: PubMedClient | None = None,
                 europepmc: EuropePmcClient | None = None, cache: TtlCache | None = None):
        self.settings = settings
        self.client = client or (llm.make_client(settings.anthropic_api_key) if settings.has_anthropic_key else None)
        self.pubmed = pubmed or PubMedClient(api_key=settings.ncbi_api_key, email=settings.entrez_email)
        self.europepmc = europepmc or EuropePmcClient()
        self.cache = cache or TtlCache(settings.cache_ttl_seconds, settings.cache_max_entries)

    # -- PICO only -----------------------------------------------------------
    def pico(self, question: str, request_id: str | None = None) -> Pico | None:
        deadline = Deadline(min(60.0, self.settings.request_deadline_seconds))
        return llm.extract_pico(self._client(), self.settings.model, question, deadline)

    # -- Full report ---------------------------------------------------------
    def answer(self, question: str, request_id: str | None = None) -> EvidenceResult:
        request_id = request_id or uuid.uuid4().hex[:12]
        key = question_key(question)
        cached = self.cache.get(key)
        if cached is not None:
            log.info("rid=%s cache hit", request_id)
            cached.request_id = request_id
            return cached

        deadline = Deadline(self.settings.request_deadline_seconds)
        timings: dict[str, float] = {}
        t = time.monotonic()

        pico = llm.extract_pico(self._client(), self.settings.model, question, deadline)
        timings["pico"] = time.monotonic() - t
        if pico is None:
            summary = report.insufficient_markdown(None, "This does not read as a clinical question the medical literature can answer. Try naming a patient group, a treatment or test, and an outcome.")
            return EvidenceResult("insufficient", None, None, [], summary, "none", timings=timings, request_id=request_id)

        t = time.monotonic()
        try:
            retrieval = retrieve(pico, pubmed=self.pubmed, europepmc=self.europepmc, deadline=deadline,
                                 min_hits=self.settings.min_hits, fetch_max=self.settings.fetch_max)
        except UpstreamError as exc:
            log.warning("rid=%s retrieval failed on both sources: %s", request_id, exc.detail)
            raise
        timings["retrieval"] = time.monotonic() - t
        ladder = [{"step": s.name, "hits": s.hits} for s in retrieval.steps_run]
        log.info("rid=%s source=%s step=%r steps=%d candidates=%d", request_id, retrieval.source,
                 retrieval.selected_step, len(retrieval.steps_run), len(retrieval.articles))

        if not retrieval.articles:
            summary = report.insufficient_markdown(pico, "No indexed abstracts matched this population and intervention. The question may be too specific, or the terms may need rewording.")
            result = EvidenceResult("insufficient", pico, None, [], summary, retrieval.source, ladder, timings, request_id)
            self.cache.set(key, result)
            return result

        t = time.monotonic()
        top = rank(retrieval.articles, pico.search_text(), top_k=self.settings.synthesis_top_k)
        timings["rank"] = time.monotonic() - t

        t = time.monotonic()
        deadline.check("synthesis")
        synthesis = llm.synthesize(self._client(), self.settings.model, pico, top, deadline)
        timings["synthesis"] = time.monotonic() - t
        if synthesis is None:
            summary = report.insufficient_markdown(pico, "The evidence summary could not be produced for this question.")
            return EvidenceResult("insufficient", pico, None, top, summary, retrieval.source, ladder, timings, request_id)

        t = time.monotonic()
        synthesis = verify(synthesis, top)
        timings["verify"] = time.monotonic() - t
        verified = [f for f in synthesis.findings if f.verified]
        log.info("rid=%s findings=%d verified=%d certainty=%s cap=%s", request_id, len(synthesis.findings),
                 len(verified), synthesis.certainty, synthesis.certainty_cap)

        summary = report.render_markdown(pico, synthesis, top, source=retrieval.source)
        status = "ok" if verified else "insufficient"
        result = EvidenceResult(status, pico, synthesis, top, summary, retrieval.source, ladder, timings, request_id)
        self.cache.set(key, result)
        return result

    def _client(self):
        if self.client is None:
            raise llm.LlmUnavailable("no Claude API key configured")
        return self.client


__all__ = ["EvidenceService", "EvidenceResult", "DeadlineExceeded", "UpstreamError"]
