from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence.config import load_settings  # noqa: E402
from evidence.pubmed import parse_efetch_xml  # noqa: E402
from evidence.util import Deadline  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def efetch_xml() -> str:
    return (FIXTURES / "efetch_sample.xml").read_text()


@pytest.fixture
def articles(efetch_xml):
    return parse_efetch_xml(efetch_xml)


@pytest.fixture
def europepmc_json() -> dict:
    return json.loads((FIXTURES / "europepmc_sample.json").read_text())


@pytest.fixture
def settings():
    return load_settings({
        "CLAUDE_API_KEY": "test-key\r\n",
        "ALLOWED_ORIGINS": "https://metacare.ai,http://localhost:3000",
        "REQUEST_DEADLINE_SECONDS": "30",
    })


@pytest.fixture
def deadline():
    return Deadline(30)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class FakeSession:
    """Scripted requests.Session: each GET pops the next scripted response, or raises."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {}), timeout))
        if not self.responses:
            raise AssertionError("unexpected extra HTTP call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_session_factory():
    return FakeSession


@pytest.fixture
def fake_response_factory():
    return FakeResponse


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Message:
    def __init__(self, payload, stop_reason="end_turn", model="claude-opus-5-5"):
        self.content = [_Block(json.dumps(payload) if not isinstance(payload, str) else payload)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.model = model


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, _Message):
            return item
        return _Message(item)


class FakeClaude:
    """Mimics anthropic.Anthropic().with_options(...).beta.messages.create(...)."""

    def __init__(self, script):
        self.messages = FakeMessages(script)
        self.beta = self

    def with_options(self, **kwargs):
        return self


@pytest.fixture
def fake_claude_factory():
    return FakeClaude


@pytest.fixture
def refusal_message_factory():
    def make(category="bio"):
        msg = _Message({}, stop_reason="refusal")
        class SD: pass
        msg.stop_details = SD(); msg.stop_details.category = category
        return msg
    return make


@pytest.fixture
def pico_payload():
    return {
        "population": {"text": "adults having total knee replacement", "synonyms": ["total knee arthroplasty", "knee replacement"], "mesh": ["Arthroplasty, Replacement, Knee"]},
        "intervention": {"text": "tranexamic acid", "synonyms": ["TXA"], "mesh": ["Tranexamic Acid"]},
        "comparison": {"text": "placebo", "synonyms": [], "mesh": []},
        "outcome": {"text": "blood transfusion", "synonyms": ["transfusion"], "mesh": ["Blood Transfusion"]},
        "question_type": "therapy",
        "rewritten_question": "In adults having total knee replacement, does tranexamic acid reduce transfusion compared with placebo?",
        "answerable": True,
    }


@pytest.fixture
def synthesis_payload():
    return {
        "bottom_line": "Tranexamic acid roughly halves the risk of transfusion after total knee arthroplasty.",
        "findings": [
            {"statement": "Tranexamic acid reduced transfusion risk (relative risk 0.45, 95% CI 0.38 to 0.53).",
             "pmids": ["30000001"],
             "quotes": [{"pmid": "30000001", "text": "Tranexamic acid reduced the risk of transfusion (relative risk 0.45, 95% CI 0.38 to 0.53)"}]},
            {"statement": "Intravenous and topical routes had similar transfusion rates (4% versus 5%).",
             "pmids": ["30000002"],
             "quotes": [{"pmid": "30000002", "text": "transfusion occurred in 4% of the intravenous group and 5% of the topical group"}]},
            {"statement": "Tranexamic acid cut transfusion by 80% in a large trial.",
             "pmids": ["30000001"],
             "quotes": [{"pmid": "30000001", "text": "Tranexamic acid reduced the risk of transfusion"}]},
            {"statement": "A fabricated finding citing an unknown study.",
             "pmids": ["99999999"],
             "quotes": [{"pmid": "99999999", "text": "something"}]},
        ],
        "certainty": "high",
        "certainty_rationale": "Pooled randomized trials with a precise estimate.",
        "limitations": ["Dosing regimens varied across trials."],
        "insufficient": False,
    }
