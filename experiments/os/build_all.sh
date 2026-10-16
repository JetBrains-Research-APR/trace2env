#!/usr/bin/env bash
# Build the 5 os cross-fit packages: ingest+extract the five folds one after another (the shared cache makes every
# transition a paid call exactly once), then induce and compile the five folds in parallel.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
: "${OPENROUTER_API_KEY:?}"
EXP="work/exp-os-cv"
style=os
for f in 0 1 2 3 4; do STAGES="ingest extract" bash "$EXP/run_reconstruction.sh" "$style-f$f" || echo "[$(date -Is)] $style-f$f EXTRACT FAILED"; done
for f in 0 1 2 3 4; do
  STAGES="schema rules renderers notes compile" bash "$EXP/run_reconstruction.sh" "$style-f$f" > "$EXP/build-$style-f$f.log" 2>&1 || echo "[$(date -Is)] $style-f$f INDUCTION FAILED" &
done
wait
echo "[$(date -Is)] BUILD_ALL DONE"
