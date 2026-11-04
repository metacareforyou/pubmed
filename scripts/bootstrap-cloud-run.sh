#!/usr/bin/env bash
# One-time creation of the two Cloud Run services for MetaCare Evidence.
# After this, the deploy workflow only swaps the image (it has no rights to
# change environment, secrets or scaling), so runtime settings live here.
#
# Usage: scripts/bootstrap-cloud-run.sh <image> [--prod] [--stage]
#   <image>  a pushed us-central1-docker.pkg.dev/.../metacare-evidence:<sha>
# Defaults to both services. Requires gcloud logged in with rights on the
# shared project, and the secret claude-api-key to hold a valid Claude key.
set -euo pipefail

IMAGE="${1:?image required}"; shift || true
PROJECT="${PROJECT:-vivid-rush-447016-d8}"
REGION="${REGION:-us-central1}"
WANT_PROD=1; WANT_STAGE=1
for arg in "$@"; do
  case "$arg" in
    --prod) WANT_STAGE=0 ;;
    --stage) WANT_PROD=0 ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done

ORIGINS="https://metacare.ai,https://www.metacare.ai,https://dev.metacare.ai:8443,http://localhost:3000"
ENV_VARS="^;^ENTREZ_EMAIL=info@metacare.ai;ALLOWED_ORIGINS=${ORIGINS};EVIDENCE_MODEL=${EVIDENCE_MODEL:-claude-opus-5-5};REQUEST_DEADLINE_SECONDS=120"
SECRETS="CLAUDE_API_KEY=claude-api-key:latest"
if [ -n "${NCBI_API_KEY_SECRET:-}" ]; then
  SECRETS="${SECRETS},NCBI_API_KEY=${NCBI_API_KEY_SECRET}:latest"
fi

deploy() {
  local service="$1" max="$2"
  echo "== ${service}"
  gcloud run deploy "${service}" \
    --project "${PROJECT}" --region "${REGION}" --platform managed \
    --image "${IMAGE}" \
    --allow-unauthenticated \
    --port 8080 --cpu 1 --memory 512Mi --concurrency 20 --timeout 180 \
    --min-instances 0 --max-instances "${max}" --cpu-boost \
    --set-env-vars "${ENV_VARS}" \
    --set-secrets "${SECRETS}" \
    --quiet
  local url
  url=$(gcloud run services describe "${service}" --project "${PROJECT}" --region "${REGION}" --format='value(status.url)')
  echo "   ${url}"
  curl -sS "${url}/healthz"; echo
}

[ "${WANT_PROD}" = 1 ] && deploy metacare-evidence 5
[ "${WANT_STAGE}" = 1 ] && deploy metacare-evidence-stage 3

cat <<MSG

Next:
  1. Merge metacare-shared-services PR #39 so the pubmed repository's deploy
     workflow can update these services, then set the repository variables
     GCP_PROJECT_ID=${PROJECT}
     GCP_WORKLOAD_IDENTITY_PROVIDER=<same as metacare-gpt-backend>
     GCP_PUBLISHER_SERVICE_ACCOUNT=<output app_publisher_emails.evidence>
  2. Smoke test: scripts/smoke.sh <url>
MSG
