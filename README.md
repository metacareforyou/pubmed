# MetaCare Evidence

Clinician-facing evidence-based medicine: a clinical question becomes a PICO
frame, a PubMed search ladder, a ranked set of abstracts and a verified,
certainty-capped evidence report. It is the successor to the Metacare.ai
PICO / PubMed feature and keeps that page's API shape.

## How a request flows

1. **PICO** (Claude, structured output): population, intervention, comparison,
   outcome, each with search synonyms and MeSH headings, plus the question type.
   Unanswerable questions stop here.
2. **Query ladder** (PubMed E-utilities): synonyms are ORed inside each element,
   population and intervention are required, comparison and outcome optional.
   Each step starts with the systematic review subset, then PubMed Clinical
   Queries filters, then no filter, and relaxes until there are enough hits.
   If NCBI fails after retries, Europe PMC answers the same search.
3. **Ranking**: BM25 relevance plus study design plus recency, top 8.
4. **Synthesis** (Claude, structured output): bottom line, findings with PMIDs
   and exact quotes, GRADE style certainty, limitations.
5. **Verification**: a finding survives only if every PMID was retrieved, every
   quote is found in that record, and every number in the statement appears in
   a cited abstract. Certainty is capped by the weakest cited design
   (observational evidence can never be reported above low).
6. **Report**: Markdown for people, JSON for machines.

Question text never reaches the logs or the cache keys (only a digest does).

## API

| Method and path | Body | Returns |
|---|---|---|
| `POST /pubmedsummary` | form or JSON `question` | legacy shape: `pico_question`, `articles` (arrays), `summary` (Markdown) |
| `POST /pico` | form or JSON `question` | `pico_question` text and a structured `pico` |
| `POST /v1/evidence` | JSON `{"question": "..."}` | full report: PICO, verified findings with quotes, certainty and its cap, ranked articles, ladder steps, timings |
| `GET /healthz` | | status, version, model |

Errors are JSON with a `request_id`. 400 for a missing question, 413 for a
long one, 503 when Claude or both literature sources are down, 504 when the
request deadline passes. The `X-Request-Id` header is echoed or generated.

## Configuration

Copy `.env.example` to `.env`. `CLAUDE_API_KEY` (or `ANTHROPIC_API_KEY`) is
required. `NCBI_API_KEY` is optional and free; it raises NCBI's limit from 3 to
10 requests a second. `ALLOWED_ORIGINS` is the browser CORS allow list.
`EVIDENCE_MODEL` defaults to `claude-opus-5-5`.

## Run

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
.venv/bin/python -m evidence            # serves on :8080 with Waitress
curl -s -X POST localhost:8080/v1/evidence -H 'content-type: application/json' \
  -d '{"question":"In adults having knee replacement, does tranexamic acid reduce transfusion?"}'
```

Docker: `docker build -t metacare-evidence . && docker run --rm -p 8080:8080 --env-file .env metacare-evidence`.
The image runs as a non-root user.

## Deploy

`develop` deploys `metacare-evidence-stage` and `main` deploys
`metacare-evidence` on Cloud Run in the shared project, through the
`deploy` GitHub Actions workflow and a publisher account defined in
`metacare-shared-services` (`app-publishers.tf`). The workflow only changes the
image; environment and secrets live on the service.

## Tests

Everything runs against fixtures; no network. `tests/fixtures/` holds a PubMed
efetch XML sample and a Europe PMC JSON sample, and the Claude client is a
scripted double, so verification and the certainty cap are exercised end to end.
