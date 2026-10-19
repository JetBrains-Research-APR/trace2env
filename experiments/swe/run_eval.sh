#!/usr/bin/env bash
# SWE cross-fit evaluation on all 472 rows (99 trajectories): every trajectory's shard runs with the package of its
# scaffold built without its fold (PKG_MAP = package_map.json; schema-only twins via PKG_MAP_SCHEMA; the raw-trace corpus
# of prompting_rag via TRACE_CORPUS_MAP). Same launcher and settings as the other environments; judge gpt-5.2 via the configured LiteLLM gateway.
# Usage: run_eval.sh gpt|deepseek   (SYSTEMS overrides the system list; default: prompting trace2env_v531)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP="work/exp-swe-cv"
BACKBONE="${1:?gpt|deepseek}"
export EXP_ROOT="$EXP" PKG_MAP=package_map.json PKG_MAP_SCHEMA=package_map_schema.json TRACE_CORPUS_MAP=corpus_map.json \
  PKG_FULL=ws/claude_code-f0/packages/claude_code-f0 PKG_SCHEMA=ws/claude_code-f0/packages/claude_code-f0-schema_only \
  TRACE_CORPUS=work/exp-swe-cv/corpora/claude_code-f0/episodes \
  RUN_NAME=full472 ROWS_FILE=./full472_rows.jsonl SHARDS_DIR=./full472_shards SYSTEMS="${SYSTEMS:-prompting trace2env_v531}"
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
