from evidence.europepmc import EuropePmcClient, build_query, parse_results
from evidence.models import Pico, PicoElement


def test_parse_skips_records_without_pmid(europepmc_json):
    out = parse_results(europepmc_json)
    assert [a.pmid for a in out] == ["30000001", "30000002"]
    assert all(a.source == "europepmc" for a in out)


def test_parse_fields(europepmc_json):
    a = parse_results(europepmc_json)[0]
    assert a.authors == ["Smith JA", "Lee K"]
    assert a.year == 2023 and a.pmcid == "PMC9000001"
    assert a.publication_types == ["Meta-Analysis", "Systematic Review"]
    assert a.mesh_terms == ["Tranexamic Acid"]


def test_bad_year_is_none(europepmc_json):
    assert parse_results(europepmc_json)[1].year is None


def test_build_query_requires_medline_source():
    pico = Pico(PicoElement("knee replacement", ["total knee arthroplasty"]), PicoElement("tranexamic acid"), PicoElement(""), PicoElement("transfusion"))
    q = build_query(pico, require_outcome=True)
    assert q.startswith('("knee replacement" OR "total knee arthroplasty") AND (tranexamic acid') or '"tranexamic acid"' in q
    assert "SRC:MED" in q and "transfusion" in q
    assert "transfusion" not in build_query(pico, require_outcome=False)


def test_client_search(fake_session_factory, fake_response_factory, europepmc_json, deadline):
    session = fake_session_factory([fake_response_factory(200, europepmc_json)])
    client = EuropePmcClient(session=session, sleep=lambda s: None)
    out = client.search("q", page_size=5, deadline=deadline)
    assert len(out) == 2
    assert session.calls[0][1]["resultType"] == "core"
