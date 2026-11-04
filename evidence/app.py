"""Flask application: legacy /pico and /pubmedsummary, new /v1/evidence and /healthz."""

from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from . import __version__
from .config import Settings, load_settings
from .llm import LlmUnavailable
from .pipeline import EvidenceService
from .util import DeadlineExceeded, UpstreamError

log = logging.getLogger("evidence")

MAX_QUESTION_CHARS = 1000


def _question_from_request() -> str | None:
    """Accept the legacy form post and JSON. Returns None when absent."""
    if request.is_json:
        body = request.get_json(silent=True) or {}
        value = body.get("question")
    else:
        value = request.form.get("question") or request.values.get("question")
    if value is None:
        return None
    return str(value).strip()


def _bad_request(message: str, status: int = 400):
    return jsonify({"error": message, "request_id": g.get("request_id", "")}), status


def create_app(settings: Settings | None = None, service: EvidenceService | None = None) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    CORS(app, origins=settings.allowed_origins, methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "X-Request-Id"], expose_headers=["X-Request-Id"])
    app.extensions["evidence_service"] = service or EvidenceService(settings)
    app.extensions["evidence_settings"] = settings

    @app.before_request
    def _start():
        g.request_id = (request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12])[:64]
        g.started = time.monotonic()

    @app.after_request
    def _finish(response):
        response.headers["X-Request-Id"] = g.get("request_id", "")
        response.headers["Cache-Control"] = "no-store"
        if request.path != "/healthz":
            log.info("rid=%s %s %s -> %d in %dms", g.get("request_id"), request.method, request.path,
                     response.status_code, int((time.monotonic() - g.get("started", time.monotonic())) * 1000))
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "version": __version__, "model": settings.model,
                        "claude_configured": settings.has_anthropic_key})

    @app.get("/")
    def index():
        return jsonify({"service": "metacare-evidence", "version": __version__,
                        "endpoints": ["/healthz", "/pico", "/pubmedsummary", "/v1/evidence"]})

    def _validated_question():
        question = _question_from_request()
        if question is None or not question:
            return None, _bad_request("Please provide a clinical question that is answerable with an evidence based medicine report.")
        if len(question) > MAX_QUESTION_CHARS:
            return None, _bad_request(f"Question is too long; keep it under {MAX_QUESTION_CHARS} characters.", 413)
        return question, None

    @app.post("/pico")
    def pico():
        question, error = _validated_question()
        if error:
            return error
        result = app.extensions["evidence_service"].pico(question, g.request_id)
        if result is None:
            return jsonify({"pico_question": "", "pico": None,
                            "message": "This does not read as a clinical question the medical literature can answer."})
        return jsonify({"pico_question": result.as_text(), "pico": result.to_dict()})

    @app.post("/pubmedsummary")
    def pubmedsummary():
        question, error = _validated_question()
        if error:
            return error
        result = app.extensions["evidence_service"].answer(question, g.request_id)
        return jsonify(result.legacy_payload())

    @app.post("/v1/evidence")
    def evidence():
        question, error = _validated_question()
        if error:
            return error
        result = app.extensions["evidence_service"].answer(question, g.request_id)
        return jsonify(result.payload())

    @app.errorhandler(LlmUnavailable)
    def _llm_down(exc):
        log.error("rid=%s llm unavailable: %s", g.get("request_id"), exc)
        return jsonify({"error": "The evidence service cannot reach its language model right now. Please try again later.",
                        "request_id": g.get("request_id", "")}), 503

    @app.errorhandler(UpstreamError)
    def _upstream_down(exc):
        log.error("rid=%s upstream unavailable: %s", g.get("request_id"), exc)
        return jsonify({"error": "PubMed and Europe PMC are not answering right now. Please try again in a few minutes.",
                        "request_id": g.get("request_id", "")}), 503

    @app.errorhandler(DeadlineExceeded)
    def _too_slow(exc):
        log.error("rid=%s deadline exceeded at %s", g.get("request_id"), exc)
        return jsonify({"error": "The search took too long. Try a narrower question.",
                        "request_id": g.get("request_id", "")}), 504

    @app.errorhandler(413)
    def _too_big(exc):
        return _bad_request("Request body too large.", 413)

    @app.errorhandler(404)
    def _not_found(exc):
        return _bad_request("Not found.", 404)

    @app.errorhandler(405)
    def _not_allowed(exc):
        return _bad_request("Method not allowed.", 405)

    @app.errorhandler(Exception)
    def _unexpected(exc):
        log.exception("rid=%s unexpected error: %s", g.get("request_id"), type(exc).__name__)
        return jsonify({"error": "Unexpected error.", "request_id": g.get("request_id", "")}), 500

    return app
