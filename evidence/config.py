"""Settings read once from the environment. No secrets are logged anywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    ncbi_api_key: str | None
    entrez_email: str
    allowed_origins: list[str]
    model: str
    request_deadline_seconds: float
    cache_ttl_seconds: float
    cache_max_entries: int
    port: int
    # How many PubMed hits a ladder step needs before the ladder stops relaxing.
    min_hits: int = 12
    # How many PMIDs to fetch per search.
    fetch_max: int = 60
    # How many ranked articles go to synthesis.
    synthesis_top_k: int = 8
    extra: dict = field(default_factory=dict)

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    key = (env.get("CLAUDE_API_KEY") or env.get("ANTHROPIC_API_KEY") or "").strip() or None
    return Settings(
        anthropic_api_key=key,
        ncbi_api_key=(env.get("NCBI_API_KEY") or "").strip() or None,
        entrez_email=env.get("ENTREZ_EMAIL", "info@metacare.ai"),
        allowed_origins=_split_csv(
            env.get(
                "ALLOWED_ORIGINS",
                "https://metacare.ai,https://www.metacare.ai,http://localhost:3000",
            )
        ),
        model=env.get("EVIDENCE_MODEL", "claude-opus-5-5"),
        request_deadline_seconds=float(env.get("REQUEST_DEADLINE_SECONDS", "90")),
        cache_ttl_seconds=float(env.get("CACHE_TTL_SECONDS", str(6 * 3600))),
        cache_max_entries=int(env.get("CACHE_MAX_ENTRIES", "256")),
        port=int(env.get("PORT", "8080")),
    )
