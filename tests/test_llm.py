import anthropic
import pytest

from evidence import llm
from evidence.models import Pico, PicoElement
from evidence.util import Deadline


def test_extract_pico_builds_structure(fake_claude_factory, pico_payload):
    client = fake_claude_factory([pico_payload])
    pico = llm.extract_pico(client, "claude-opus-5-5", "q", Deadline(30))
    assert pico.population.text == "adults having total knee replacement"
    assert pico.intervention.all_terms() == ["tranexamic acid", "TXA"]
    assert pico.question_type == "therapy"
    call = client.messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and llm.FALLBACK_BETA in call["betas"]
    assert "temperature" not in call and "thinking" not in call


def test_extract_pico_returns_none_when_not_answerable(fake_claude_factory, pico_payload):
    pico_payload["answerable"] = False
    assert llm.extract_pico(fake_claude_factory([pico_payload]), "m", "q", Deadline(30)) is None


def test_extract_pico_returns_none_without_intervention(fake_claude_factory, pico_payload):
    pico_payload["intervention"]["text"] = ""
    assert llm.extract_pico(fake_claude_factory([pico_payload]), "m", "q", Deadline(30)) is None


def test_refusal_returns_none_not_error(fake_claude_factory, refusal_message_factory):
    assert llm.extract_pico(fake_claude_factory([refusal_message_factory()]), "m", "q", Deadline(30)) is None


def test_auth_error_becomes_llm_unavailable(fake_claude_factory):
    import httpx2
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad", response=httpx2.Response(401, request=req), body=None)
    with pytest.raises(llm.LlmUnavailable):
        llm.extract_pico(fake_claude_factory([err]), "m", "q", Deadline(30))


def test_billing_400_is_unavailable_with_reason(fake_claude_factory):
    import httpx2
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API."}}
    err = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body=body)
    with pytest.raises(llm.LlmUnavailable) as info:
        llm.extract_pico(fake_claude_factory([err]), "m", "q", Deadline(30))
    assert "400 invalid_request_error: Your credit balance is too low" in str(info.value)


def test_synthesize_parses_findings(fake_claude_factory, synthesis_payload, articles):
    pico = Pico(PicoElement("p"), PicoElement("i"), PicoElement(""), PicoElement("o"))
    s = llm.synthesize(fake_claude_factory([synthesis_payload]), "m", pico, articles[:2], Deadline(30))
    assert len(s.findings) == 4
    assert s.findings[0].quotes[0].pmid == "30000001"
    assert s.stated_certainty == "high"
    user = fake_claude_factory.__name__ and None  # silence linters
    prompt = s and None


def test_synthesis_prompt_lists_pmids_and_designs(fake_claude_factory, synthesis_payload, articles):
    client = fake_claude_factory([synthesis_payload])
    pico = Pico(PicoElement("p"), PicoElement("i"), PicoElement(""), PicoElement("o"))
    articles[0].design = "systematic_review"
    llm.synthesize(client, "m", pico, articles[:1], Deadline(30))
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "PMID 30000001" in content and "design: systematic_review" in content
    assert client.messages.calls[0]["output_config"]["effort"] == "medium"


def test_non_json_text_returns_none(fake_claude_factory):
    assert llm.extract_pico(fake_claude_factory(["not json"]), "m", "q", Deadline(30)) is None
