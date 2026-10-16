#!/usr/bin/env bash
# OS cross-fit evaluation on all 200 rows (77 trajectories): every trajectory's shard runs with the package built without
# its fold (PKG_MAP = package_map.json). Same launcher and settings as the other environments; judge gpt-5.2 via OpenRouter.
# Usage: run_eval.sh gpt|deepseek   (SYSTEMS overrides the system list; default: trace2env_v531 only)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
: "${OPENROUTER_API_KEY:?}"
EXP="work/exp-os-cv"
BACKBONE="${1:?gpt|deepseek}"
export EXP_ROOT="$EXP" PKG_MAP=package_map.json PKG_MAP_SCHEMA=package_map_schema.json TRACE_CORPUS_MAP=corpus_map.json \
  PKG_FULL=ws/os-f0/packages/os-f0 PKG_SCHEMA=ws/os-f0/packages/os-f0-schema_only \
  TRACE_CORPUS=work/exp-os-cv/corpora/os-f0/episodes \
  RUN_NAME=full200 ROWS_FILE=./full200_rows.jsonl SHARDS_DIR=./full200_shards SYSTEMS="${SYSTEMS:-trace2env_v531}"
case "$BACKBONE" in
  gpt)
    BACKBONE_MODEL=openai/gpt-5.6-sol BACKBONE_SLUG=gpt-5.6-sol REASONING_ARGS="--chat-reasoning-effort medium" AGENT_MAX_OUTPUT_TOKENS=16384 MAX_PARALLEL=16 \
      bash experiments/terminal/backbones/run_heldout_systems.sh ;;
  deepseek)
    : "${DEEPSEEK_API_KEY:?}"
    BACKBONE_MODEL=deepseek-flash BACKBONE_SLUG=deepseek-flash-official BASE_URL=https://api.deepseek.com API_KEY_ENV=DEEPSEEK_API_KEY \
      REASONING_ARGS="--chat-reasoning-effort low" AGENT_MAX_OUTPUT_TOKENS=65536 SCHEMA_ARGS="--chat-schema-mode json_object" AGENT_ARGS="--agent-transport tools" MAX_PARALLEL=8 \
      bash experiments/terminal/backbones/run_heldout_systems.sh ;;
  *) echo "unknown backbone $BACKBONE"; exit 2 ;;
esac
