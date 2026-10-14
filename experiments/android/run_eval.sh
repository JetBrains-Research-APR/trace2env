#!/usr/bin/env bash
# Android cross-fit evaluation on all 200 rows (92 trajectories): every trajectory's shard runs with the package of its
# sub-source built without its fold (PKG_MAP = package_map.json). Same launcher and settings as the terminal, MCP and
# web evaluations (work/exp-v1_20_r=1/backbones/run_heldout_systems.sh); the judge is gpt-5.2 via the configured LiteLLM gateway.
# Usage: run_eval.sh gpt|deepseek   (SYSTEMS overrides the system list; default: prompting trace2env_v533; the baselines:
#   SYSTEMS="harness_only_v51 envpack_prompting prompting_rag" — schema-only twins via PKG_MAP_SCHEMA, per-trajectory raw-trace
#   corpora via TRACE_CORPUS_MAP)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP="work/exp-android-cv"
BACKBONE="${1:?gpt|deepseek}"
export EXP_ROOT="$EXP" PKG_MAP=package_map.json PKG_MAP_SCHEMA=package_map_schema.json TRACE_CORPUS_MAP=corpus_map.json \
  PKG_FULL=ws/json-f0/packages/json-f0 PKG_SCHEMA=ws/json-f0/packages/json-f0-schema_only TRACE_CORPUS=work/exp-android-cv/corpora/json-f0/episodes \
  RUN_NAME=full200 ROWS_FILE=./full200_rows.jsonl SHARDS_DIR=./full200_shards SYSTEMS="${SYSTEMS:-prompting trace2env_v533}"
case "$BACKBONE" in
  gpt)
    BACKBONE_MODEL=openai/gpt-6.1-sol BACKBONE_SLUG=gpt-6.1-sol REASONING_ARGS="--chat-reasoning-effort medium" AGENT_MAX_OUTPUT_TOKENS=16384 MAX_PARALLEL=16 \
      bash experiments/terminal/backbones/run_heldout_systems.sh ;;
  deepseek)

    BACKBONE_MODEL="${DEEPSEEK_MODEL:?set DEEPSEEK_MODEL to a model ID available in LiteLLM}" BACKBONE_SLUG=deepseek-flash-official \
      REASONING_ARGS="--chat-reasoning-effort low" AGENT_MAX_OUTPUT_TOKENS=65536 SCHEMA_ARGS="--chat-schema-mode json_object" AGENT_ARGS="--agent-transport tools" MAX_PARALLEL=8 \
      bash experiments/terminal/backbones/run_heldout_systems.sh ;;
  *) echo "unknown backbone $BACKBONE"; exit 2 ;;
esac
