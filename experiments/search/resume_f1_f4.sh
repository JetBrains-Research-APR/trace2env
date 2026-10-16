#!/usr/bin/env bash
# Resume after the U+2028 JSONL reader fix: re-ingest and extract folds 1-4 in sequence (shared cache), then induce them in parallel.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
: "${OPENROUTER_API_KEY:?}"
EXP="work/exp-search-cv"
for f in 1 2 3 4; do STAGES="ingest extract" bash "$EXP/run_reconstruction.sh" "search-f$f" || echo "[$(date -Is)] search-f$f EXTRACT FAILED"; done
for f in 1 2 3 4; do
  STAGES="schema rules renderers notes compile" bash "$EXP/run_reconstruction.sh" "search-f$f" > "$EXP/build-search-f$f.resume.log" 2>&1 || echo "[$(date -Is)] search-f$f INDUCTION FAILED" &
done
wait
echo "[$(date -Is)] RESUME DONE"
