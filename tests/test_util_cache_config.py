import pytest

from evidence.cache import TtlCache, question_key
from evidence.config import load_settings
from evidence.util import Deadline, DeadlineExceeded, RateLimiter


def test_settings_strip_key_whitespace_and_prefer_claude_key(settings):
    assert settings.anthropic_api_key == "test-key"
    s = load_settings({"ANTHROPIC_API_KEY": "a", "CLAUDE_API_KEY": "c"})
    assert s.anthropic_api_key == "c"
    assert load_settings({}).model == "claude-opus-5-5"


def test_settings_origins_split(settings):
    assert settings.allowed_origins == ["https://metacare.ai", "http://localhost:3000"]


def test_deadline_counts_down_and_raises():
    now = [100.0]
    d = Deadline(10, clock=lambda: now[0])
    assert d.remaining() == 10 and d.timeout(30) == 10
    now[0] = 111
    assert d.expired()
    with pytest.raises(DeadlineExceeded):
        d.check("x")


def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept = []
    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=lambda s: slept.append(s))
    limiter.wait(); limiter.wait(); limiter.wait()
    assert slept == [0.5, 1.0]


def test_cache_key_is_a_digest_of_the_normalized_question():
    assert question_key("A  b") == question_key("a b")
    assert "a b" not in question_key("a b") and len(question_key("x")) == 64


def test_cache_ttl_and_eviction():
    now = [0.0]
    cache = TtlCache(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    cache.set("a", 1); cache.set("b", 2); cache.set("c", 3)
    assert cache.get("a") is None and cache.get("c") == 3
    now[0] = 11
    assert cache.get("c") is None
