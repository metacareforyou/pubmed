import pytest

from evidence.app import create_app
from evidence.llm import LlmUnavailable
from evidence.pipeline import EvidenceService
from evidence.util import DeadlineExceeded, UpstreamError
from tests.test_ladder import ScriptedEuropePmc, ScriptedPubMed


@pytest.fixture
def client(settings, fake_claude_factory, pico_payload, synthesis_payload, articles):
    claude = fake_claude_factory([pico_payload, synthesis_payload, pico_payload])
    pmids = [a.pmid for a in articles]
    pm = ScriptedPubMed([(len(pmids), pmids)] * 20, articles)
    service = EvidenceService(settings, client=claude, pubmed=pm, europepmc=ScriptedEuropePmc([]))
    return create_app(settings, service).test_client()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok" and r.get_json()["claude_configured"] is True


def test_legacy_form_post_shape(client):
    r = client.post("/pubmedsummary", data={"question": "Does TXA reduce transfusion?"},
                    content_type="application/x-www-form-urlencoded")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"pico_question", "articles", "summary"}
    assert isinstance(body["articles"][0][3], list)
    assert body["summary"].startswith("## Bottom line")


def test_v1_evidence_json_post(client):
    r = client.post("/v1/evidence", json={"question": "Does TXA reduce transfusion?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["findings"] and body["pico"]["intervention"]["text"] == "tranexamic acid"
    assert r.headers["X-Request-Id"]


def test_pico_endpoint(client):
    r = client.post("/pico", data={"question": "Does TXA reduce transfusion?"})
    assert r.status_code == 200
    assert r.get_json()["pico_question"].startswith("P: ")


def test_missing_question_is_400(client):
    assert client.post("/pubmedsummary", data={}).status_code == 400
    assert client.post("/v1/evidence", json={}).status_code == 400


def test_too_long_question_is_413(client):
    assert client.post("/v1/evidence", json={"question": "x" * 1001}).status_code == 413


def test_request_id_is_echoed(client):
    r = client.get("/healthz", headers={"X-Request-Id": "abc123"})
    assert r.headers["X-Request-Id"] == "abc123"


def test_cors_allow_list(client):
    ok = client.options("/pubmedsummary", headers={"Origin": "https://metacare.ai", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "https://metacare.ai"
    bad = client.options("/pubmedsummary", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert bad.headers.get("Access-Control-Allow-Origin") is None


def test_unknown_route_is_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.get_json()["error"]


def _failing_app(settings, exc):
    class Broken:
        def answer(self, q, rid=None):
            raise exc
        def pico(self, q, rid=None):
            raise exc
    return create_app(settings, Broken()).test_client()


def test_llm_unavailable_is_503(settings):
    r = _failing_app(settings, LlmUnavailable("no key")).post("/v1/evidence", json={"question": "q"})
    assert r.status_code == 503 and "language model" in r.get_json()["error"]


def test_upstream_down_is_503(settings):
    r = _failing_app(settings, UpstreamError("pubmed", "down")).post("/pubmedsummary", data={"question": "q"})
    assert r.status_code == 503


def test_deadline_is_504(settings):
    r = _failing_app(settings, DeadlineExceeded("synthesis")).post("/pubmedsummary", data={"question": "q"})
    assert r.status_code == 504


def test_unexpected_error_is_json_500_without_stack(settings):
    r = _failing_app(settings, RuntimeError("boom")).post("/pubmedsummary", data={"question": "q"})
    assert r.status_code == 500 and r.get_json() == {"error": "Unexpected error.", "request_id": r.headers["X-Request-Id"]}


def test_no_claude_key_is_503(fake_claude_factory, articles):
    from evidence.config import load_settings
    s = load_settings({"ALLOWED_ORIGINS": "https://metacare.ai"})
    assert not s.has_anthropic_key
    service = EvidenceService(s, pubmed=ScriptedPubMed([], articles), europepmc=ScriptedEuropePmc([]))
    r = create_app(s, service).test_client().post("/pico", data={"question": "q"})
    assert r.status_code == 503
