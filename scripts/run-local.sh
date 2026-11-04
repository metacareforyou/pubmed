#!/usr/bin/env bash
# Build and run MetaCare Evidence in local Docker.
#
#   scripts/run-local.sh                 # key from $CLAUDE_API_KEY in your shell
#   scripts/run-local.sh --from-secret   # key read from Secret Manager (claude-api-key)
#   scripts/run-local.sh --stop
#
# The key goes straight into the container environment; it is never written to disk
# or printed. Then: scripts/smoke.sh http://localhost:${PORT:-8090}
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-metacare-evidence-local}
PORT=${PORT:-8090}
PROJECT=${GCP_PROJECT_ID:-vivid-rush-447016-d8}
MODEL=${EVIDENCE_MODEL:-claude-opus-5-5}
ORIGINS=${ALLOWED_ORIGINS:-http://localhost:3000,https://metacare.ai,https://www.metacare.ai}

if [[ "${1:-}" == "--stop" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME was not running"
  exit 0
fi

if [[ "${1:-}" == "--from-secret" ]]; then
  key=$(gcloud secrets versions access latest --secret=claude-api-key --project="$PROJECT")
else
  key=${CLAUDE_API_KEY:-${ANTHROPIC_API_KEY:-}}
  if [[ -z "$key" ]]; then
    echo "Set CLAUDE_API_KEY in your shell or pass --from-secret." >&2
    exit 1
  fi
fi

docker build -q -t metacare-evidence:local . >/dev/null
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8080" \
  --env-file <(printf 'CLAUDE_API_KEY=%s\n' "$key") \
  -e "ALLOWED_ORIGINS=$ORIGINS" \
  -e "EVIDENCE_MODEL=$MODEL" \
  -e ENTREZ_EMAIL=info@metacare.ai \
  -e REQUEST_DEADLINE_SECONDS=120 \
  ${NCBI_API_KEY:+-e "NCBI_API_KEY=$NCBI_API_KEY"} \
  metacare-evidence:local >/dev/null
unset key

for _ in $(seq 1 20); do
  if curl -fsS "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
    echo "metacare-evidence is up on http://localhost:$PORT (container $NAME)"
    echo
    echo "Try one question:"
    echo "  curl -s -X POST http://localhost:$PORT/v1/evidence -H 'Content-Type: application/json' \\"
    echo "    -d '{\"question\":\"Does tranexamic acid reduce transfusion after total knee arthroplasty?\"}' | jq '{status,certainty,bottom_line,findings:(.findings|length)}'"
    echo
    echo "Full acceptance run (20 questions, p95 under 30 s):"
    echo "  scripts/smoke.sh http://localhost:$PORT"
    echo
    echo "Logs: docker logs -f $NAME      Stop: scripts/run-local.sh --stop"
    exit 0
  fi
  sleep 1
done
echo "service did not become healthy; docker logs $NAME" >&2
docker logs "$NAME" >&2 || true
exit 1
