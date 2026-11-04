from evidence.pubmed import PubMedClient, parse_efetch_xml
from evidence.util import Deadline, UpstreamError
import pytest


def test_parses_every_record_with_a_pmid(articles):
    assert [a.pmid for a in articles] == ["30000001", "30000002", "30000003", "30000004", "30000005"]


def test_labelled_abstract_sections_are_joined(articles):
    a = articles[0]
    assert a.abstract.startswith("Background: Blood loss")
    assert "Results: Tranexamic acid reduced the risk of transfusion" in a.abstract


def test_metadata_fields(articles):
    a = articles[0]
    assert a.title == "Tranexamic acid in total knee arthroplasty: a systematic review and meta-analysis of randomized trials"
    assert a.journal == "J Arthroplasty"
    assert a.authors == ["Smith JA", "Lee K"]
    assert a.year == 2023 and a.date == "2023 Mar"
    assert "Meta-Analysis" in a.publication_types
    assert a.mesh_terms == ["Tranexamic Acid", "Arthroplasty, Replacement, Knee"]
    assert a.keywords == ["transfusion", "TXA"]
    assert a.doi == "10.1000/j.arth.2023.01" and a.pmcid == "PMC9000001"


def test_medline_date_and_collective_author(articles):
    a = articles[2]
    assert a.year == 2010
    assert a.authors == ["Regional Knee Registry Group"]


def test_record_without_abstract_is_empty_string(articles):
    assert articles[4].abstract == ""


def test_legacy_tuple_shape(articles):
    row = articles[0].to_legacy_tuple()
    assert len(row) == 7
    assert isinstance(row[3], list) and row[3] == ["Smith JA", "Lee K"]
    assert row[5] == "transfusion, TXA"


def test_search_parses_count_and_ids(fake_session_factory, fake_response_factory, deadline):
    session = fake_session_factory([fake_response_factory(200, {"esearchresult": {"count": "123", "idlist": ["1", "2"]}})])
    client = PubMedClient(api_key="k", email="e@x", session=session, sleep=lambda s: None)
    count, ids = client.search("x", retmax=10, deadline=deadline)
    assert (count, ids) == (123, ["1", "2"])
    url, params, _ = session.calls[0]
    assert url.endswith("/esearch.fcgi")
    assert params["api_key"] == "k" and params["sort"] == "relevance" and params["db"] == "pubmed"


def test_search_retries_on_429_then_succeeds(fake_session_factory, fake_response_factory, deadline):
    session = fake_session_factory([
        fake_response_factory(429), fake_response_factory(503),
        fake_response_factory(200, {"esearchresult": {"count": "1", "idlist": ["9"]}}),
    ])
    client = PubMedClient(api_key=None, email="e@x", session=session, sleep=lambda s: None)
    assert client.search("x", retmax=5, deadline=deadline)[1] == ["9"]
    assert len(session.calls) == 3


def test_search_gives_up_after_three_failures(fake_session_factory, fake_response_factory, deadline):
    session = fake_session_factory([fake_response_factory(500)] * 3)
    client = PubMedClient(api_key=None, email="e@x", session=session, sleep=lambda s: None)
    with pytest.raises(UpstreamError) as exc:
        client.search("x", retmax=5, deadline=deadline)
    assert "pubmed" in str(exc.value)


def test_search_does_not_retry_a_400(fake_session_factory, fake_response_factory, deadline):
    session = fake_session_factory([fake_response_factory(400)])
    client = PubMedClient(api_key=None, email="e@x", session=session, sleep=lambda s: None)
    with pytest.raises(UpstreamError):
        client.search("x", retmax=5, deadline=deadline)
    assert len(session.calls) == 1


def test_fetch_with_no_ids_makes_no_call(fake_session_factory, deadline):
    session = fake_session_factory([])
    client = PubMedClient(api_key=None, email="e@x", session=session)
    assert client.fetch([], deadline=deadline) == []
