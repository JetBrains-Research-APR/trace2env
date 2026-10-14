# LiteLLM configuration and usage

Trace2Env's model clients, collectors, experiment launchers, and external-harness
wrappers use the internal LiteLLM gateway. The frozen `baseline/Word2World`
submodule is unchanged.

Install the project with `pip install -e .` and configure the repository-root `.env`:

```dotenv
LITELLM_API_BASE=https://YOUR-INTERNAL-GATEWAY/v1
LITELLM_API_KEY=YOUR-INTERNAL-KEY
```

Process environment variables with these names override the file. Credentials
are read by `src/trace2env/llm_config.py`; missing settings fail without falling
back to a public provider. The old `--base-url`, `--api-key-env`, and judge
equivalents were removed. `.env` does not select the model.

## Defaults and overrides

The default generation model is `openai/gpt-6.1-sol`. All non-judge adapters and
collectors send temperature `1.0` by default, including Responses API calls.
The CLI judge keeps `openai/gpt-5.2-2025-12-11` and temperature `0.6`; experiment
judge launchers retain their existing GPT-5.2 model IDs and temperature.

| Calls | Model override | Temperature override |
|---|---|---|
| Structured calls and world-model agents | `--model` | `--structured-temperature` |
| Prompting baselines | `--model` | `--temperature` |
| EnvScaler long-horizon task agent | `--task-model` (falls back to `--model`) | `--task-temperature` |
| Judge | `--judge-model` | `--temperature` |

ALFWorld runners also expose separate `--task-model` / `--wm-model` or
`--agent-model` / `--wm-model` options. Shell overrides depend on the launcher:
`MODEL`, `BACKBONE_MODEL`, `JUDGE_MODEL`, `TB2_MODEL`, or `DEEPSEEK_MODEL`.
Some experiment launchers still pin model arguments and historical run labels.
Use the CLI directly for arbitrary models, with a fresh output directory.

```bash
python -m trace2env awb-run rows.jsonl --mode prompting --output work/new-run/pred.jsonl
python -m trace2env awb-judge --predictions work/new-run/pred.jsonl --output work/new-run/judged.jsonl
```

## Conversion details

- Each request sends its selected LiteLLM model ID unchanged; call roles do not select models.
- Structured calls default to Responses with reasoning effort `medium`. For another model family,
  try `--provider chat`; `--chat-reasoning-effort off` omits reasoning effort on structured/tool chat calls.
- Chat structured output supports `--chat-schema-mode json_schema` or `json_object`.
  Native `--agent-transport tools` requires function-calling support.
- Existing output/context limits are not detected from the model; adjust them when changing models.
- Cache identity includes the model, sampling settings, transport, and gateway URL.
  Existing experiment outputs can still be skipped by resume logic.
- External harnesses use `python -m trace2env.llm_config run COMMAND ...` to receive the
  gateway URL/key through their OpenAI environment variables. Harbor additionally receives its API base
  and temperature explicitly from `scripts/collect_tb2.sh`.
