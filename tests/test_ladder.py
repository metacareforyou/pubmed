import pytest

from evidence.ladder import build_steps, element_clause, retrieve
from evidence.models import Pico, PicoElement
from evidence.util import UpstreamError


def pico(comparison="placebo", outcome="transfusion"):
    return Pico(
        PicoElement("knee replacement", ["total knee arthroplasty"], ["Arthroplasty, Replacement, Knee"]),
        PicoElement("tranexamic acid", ["TXA"]),
        PicoElement(comparison),
        PicoElement(outcome),
    )


def test_element_clause_ors_synonyms_and_mesh():
    assert element_clause(["knee replacement", "TKA"], ["Arthroplasty, Replacement, Knee"]) == \
        '("knee replacement"[tiab] OR TKA[tiab] OR "Arthroplasty, Replacement, Knee"[mh])'


def test_element_clause_empty_is_empty_not_parentheses():
    assert element_clause([], []) == ""


def test_steps_require_p_and_i_only():
    steps = build_steps(pico(comparison="", outcome=""))
    assert all("placebo" not in s.query for s in steps)
    assert all("()" not in s.query for s in steps)
    assert steps[0].name.startswith("PI ")


def test_steps_start_strict_and_relax():
    names = [s.name for s in build_steps(pico())]
    assert names[0] == "PICO systematic reviews"
    assert "PIO systematic reviews" in names and names[-1] == "PI any design, any language"
    assert names.index("PICO systematic reviews") < names.index("PI clinical queries broad") < names.index("PI any design")


def test_steps_raise_without_population():
    with pytest.raises(ValueError):
        build_steps(Pico(PicoElement(""), PicoElement("x"), PicoElement(""), PicoElement("")))


class ScriptedPubMed:
    def __init__(self, hits, articles):
        self.hits = list(hits)
        self.articles = articles
        self.queries = []
        self.fetched = None

    def search(self, query, *, retmax, deadline):
        self.queries.append(query)
        if not self.hits:
            raise UpstreamError("pubmed", "gave up")
        count, ids = self.hits.pop(0)
        return count, ids

    def fetch(self, pmids, *, deadline):
        self.fetched = list(pmids)
        return [a for a in self.articles if a.pmid in pmids]


class ScriptedEuropePmc:
    def __init__(self, articles):
        self.articles = articles
        self.queries = []

    def search(self, query, *, page_size, deadline):
        self.queries.append(query)
        return self.articles


def test_ladder_stops_at_first_step_with_enough_hits(articles, deadline):
    pm = ScriptedPubMed([(2, ["30000001"]), (40, ["30000001", "30000002", "30000003"])], articles)
    result = retrieve(pico(), pubmed=pm, europepmc=ScriptedEuropePmc([]), deadline=deadline, min_hits=3, fetch_max=50)
    assert len(result.steps_run) == 2
    assert result.selected_step == "PICO clinical queries narrow"
    assert result.source == "pubmed"
    assert pm.fetched == ["30000001", "30000002", "30000003"]


def test_ladder_drops_records_without_abstract(articles, deadline):
    pm = ScriptedPubMed([(50, ["30000005", "30000001"])], articles)
    result = retrieve(pico(), pubmed=pm, europepmc=ScriptedEuropePmc([]), deadline=deadline, min_hits=1, fetch_max=50)
    assert [a.pmid for a in result.articles] == ["30000001"]


def test_ladder_falls_back_to_europepmc_when_pubmed_fails(articles, deadline):
    pm = ScriptedPubMed([], articles)
    ep = ScriptedEuropePmc(articles[:2])
    result = retrieve(pico(), pubmed=pm, europepmc=ep, deadline=deadline, min_hits=1, fetch_max=50)
    assert result.source == "europepmc"
    assert result.fallback_reason == "gave up"
    assert len(result.articles) == 2
