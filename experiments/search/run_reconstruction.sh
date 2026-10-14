#!/usr/bin/env bash
# Stage-by-stage reconstruction of one search cross-fit package (CORPUS = search-f<fold>) from the corpus exported by
# make_folds.py (the trajectories outside the fold). Same model settings as the swe and android cross-fits:
# openai/gpt-6.1-sol via the configured LiteLLM gateway, chat provider, reasoning medium, 600000-byte prompt budget, 65536 output tokens
# for induction and 16384 for extraction, induction batch 100. The cache is shared by every corpus of the experiment
# (content-addressed on the prompt, role, provider and model): a transition is extracted once for the whole cross-fit
# as long as the corpora run one after another. Uses LiteLLM settings from .env.
# Usage: run_reconstruction.sh CORPUS [STAGES="ingest extract schema rules renderers notes compile"]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
CORPUS="${1:?corpus name, e.g. search-f0}"
STYLE="${CORPUS%%-f*}"
EXP="work/exp-search-cv"
WS="$EXP/ws/$CORPUS"
OR=(--description "@$EXP/descriptions/$STYLE.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model openai/gpt-6.1-sol
    --provider chat --chat-reasoning-effort medium
    --induction-batch-size 100 --cache-dir "$EXP/cache")
STAGES="${STAGES:-ingest extract schema rules renderers notes compile}"
for stage in $STAGES; do
  echo "[$(date -Is)] $CORPUS $stage"
  case "$stage" in
    ingest)
      python scripts/pilot_stages.py "$WS" ingest --episodes "$EXP/corpora/$CORPUS/episodes" --split-manifest "$EXP/corpora/$CORPUS/split_manifest.json" \
        --environment-id "awb-search" --name "AgentWorldBench search, cross-fit $CORPUS" --domain search "${OR[@]}" \
        --note "search cross-fit $CORPUS: the trajectories of the search split outside fold ${CORPUS##*-f} (longest visible prefixes)" ;;
    extract)
      python scripts/pilot_stages.py "$WS" extract --workers 12 "${OR[@]}" --max-output-tokens 16384 --note "search cross-fit $CORPUS extract (16k output cap)" ;;
    compile)
      python scripts/pilot_stages.py "$WS" compile --package-label "$CORPUS" "${OR[@]}" --note "search cross-fit $CORPUS compile" ;;
    *)
      python scripts/pilot_stages.py "$WS" "$stage" "${OR[@]}" --note "search cross-fit $CORPUS $stage" ;;
  esac
done
echo "[$(date -Is)] $CORPUS RECONSTRUCTION DONE"
