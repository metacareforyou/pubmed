import pytest

from evidence.pipeline import EvidenceService
from evidence.util import UpstreamError
from tests.test_ladder import ScriptedEuropePmc, ScriptedPubMed


def make_service(settings, claude, articles, hits=None):
    pmids = [a.pmid for a in articles]
    pm = ScriptedPubMed(hits if hits is not None else [(len(pmids), pmids)] * 20, articles)
    return EvidenceService(settings, client=claude, pubmed=pm, europepmc=ScriptedEuropePmc([])), pm


def test_full_answer_keeps_only_verified_findings(settings, fake_claude_factory, pico_payload, synthesis_payload, articles):
    service, _ = make_service(settings, fake_claude_factory([pico_payload, synthesis_payload]), articles)
    result = service.answer("Does TXA reduce transfusion in TKA?")
    assert result.status == "ok"
    verified = [f for f in result.synthesis.findings if f.verified]
    assert [f.pmids for f in verified] == [["30000001"], ["30000002"]]
    assert result.synthesis.certainty == "high"
    payload = result.payload()
    assert payload["dropped_findings"] == 2 and len(payload["findings"]) == 2
    assert payload["retrieval"]["source"] == "pubmed"
    assert "## Verified findings" in result.summary and "PMID 30000001" in result.summary


def test_legacy_payload_shape(settings, fake_claude_factory, pico_payload, synthesis_payload, articles):
    service, _ = make_service(settings, fake_claude_factory([pico_payload, synthesis_payload]), articles)
    legacy = service.answer("q").legacy_payload()
    assert set(legacy) == {"pico_question", "articles", "summary"}
    assert legacy["pico_question"].startswith("P: adults having total knee replacement")
    assert isinstance(legacy["articles"][0], list) and isinstance(legacy["articles"][0][3], list)


def test_answer_is_cached_by_normalized_question(settings, fake_claude_factory, pico_payload, synthesis_payload, articles):
    claude = fake_claude_factory([pico_payload, synthesis_payload])
    service, pm = make_service(settings, claude, articles)
    service.answer("Does TXA reduce transfusion?")
    service.answer("  does txa   reduce TRANSFUSION? ")
    assert len(claude.messages.calls) == 2
    assert len(pm.queries) >= 1


def test_unanswerable_question_is_insufficient_without_search(settings, fake_claude_factory, pico_payload, articles):
    pico_payload["answerable"] = False
    service, pm = make_service(settings, fake_claude_factory([pico_payload]), articles)
    result = service.answer("what is the meaning of life")
    assert result.status == "insufficient" and result.pico is None
    assert pm.queries == []
    assert "Insufficient" in result.summary


def test_no_hits_is_insufficient_not_error(settings, fake_claude_factory, pico_payload, articles):
    hits = [(0, [])] * 20
    service, _ = make_service(settings, fake_claude_factory([pico_payload]), articles, hits=hits)
    result = service.answer("q")
    assert result.status == "insufficient" and result.articles == []
    assert "No indexed abstracts" in result.summary


def test_nothing_verified_is_insufficient(settings, fake_claude_factory, pico_payload, synthesis_payload, articles):
    synthesis_payload["findings"] = [synthesis_payload["findings"][3]]
    service, _ = make_service(settings, fake_claude_factory([pico_payload, synthesis_payload]), articles)
    result = service.answer("q")
    assert result.status == "insufficient"
    assert result.payload()["certainty"] == "insufficient"
    assert "do not answer this question" in result.summary


def test_both_sources_down_raises_upstream(settings, fake_claude_factory, pico_payload, articles):
    class DownEurope:
        def search(self, *a, **k):
            raise UpstreamError("europepmc", "down")
    pm = ScriptedPubMed([], articles)
    service = EvidenceService(settings, client=fake_claude_factory([pico_payload]), pubmed=pm, europepmc=DownEurope())
    with pytest.raises(UpstreamError):
        service.answer("q")
