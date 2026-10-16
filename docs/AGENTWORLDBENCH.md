# Evaluating on AgentWorldBench

[AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench) (Qwen-AgentWorld) evaluates a language world
model one step at a time: a record gives the real prefix of an agent trajectory and the current action, and the world
model predicts the environment's next observation. An LLM judge scores Format, Factuality, Consistency, Realism and
Quality from 1 to 5; the official scorer rescales each to 0–100 and averages the five. `trace2env` ports the official
input layout, judge prompts (verbatim, under `src/trace2env/agentworld_prompts/`) and aggregation, and adds the harness
as a world model.

## Data

Download the test files of the dataset into `work/agentworldbench/` (`terminal_test.jsonl`, `swe_test.jsonl`,
`android_test.jsonl`, `web_test.jsonl`, ...). The files are never modified. A trajectory contributes several records,
one per evaluated turn, so statistics must treat the trajectory as the unit of dependence. The Android file reuses some
identifiers for distinct trajectories; the cross-fit tooling under `experiments/android/` derives unique ones.

## Commands

```bash
# Benchmark trajectories as construction episodes (smoke tests and cross-fits; a SplitManifest is written)
python -m trace2env awb-export work/agentworldbench/terminal_test.jsonl --output work/awb/train

# Predict every record with one of the four systems (see the table), then judge and score
python -m trace2env awb-run work/agentworldbench/terminal_test.jsonl --mode agentic --package WORLDBOOK \
  --official-input --knowledge-gate --features default,evidence --allow-unvalidated --output work/awb/pred.jsonl \
  --provider chat --base-url URL --api-key-env KEY_VAR --model MODEL --cache-dir work/awb/cache
python -m trace2env awb-judge --predictions work/awb/pred.jsonl --output work/awb/judged.jsonl \
  --judge-model gpt-5.2 --judge-api-key-env KEY_VAR --cache-dir work/awb/judge-cache
python -m trace2env awb-score --predictions work/awb/judged.jsonl --summary work/awb/score.json
```

| `--mode` | What is predicted from | Use |
|---|---|---|
| `prompting` | The record's own system prompt and full history in one chat call, exactly as the official inference code builds it | The official baseline (Direct Prompting) |
| `prompting_rag` | The official input plus `--rag-top-k` raw turns retrieved lexically from `--trace-corpus` (a directory of construction traces) | Trace RAG Prompting |
| `envpack_prompting` | The official input plus a fixed, non-agentic view of `--package` (schemas, rules, contracts, notes, demonstrations, retrieved evidence) | Worldbook Prompting |
| `agentic` | The harness: the prefix is reconstructed into session state and episodic memory, and the world-model agent simulates the turn with its tools; `--no-state-tracking --no-package-knowledge` on a schema-only twin of the worldbook gives the package-free control (Harness only) | Trace2Env |

Output rows keep every benchmark field plus `gen` (the `<predicted_observation>`-wrapped prediction) and a
`trace2env` field with the route (`rule`, `planner`, `fallback_single_shot`, `fallback`, `error`), tool calls, usage and
gate decisions, so the official `eval.py judge` and `score` also accept them. Records the judge cannot score are excluded,
as in the official scorer. Pass `--cache-dir` to both commands: calls are content-addressed and an interrupted run resumes.

## Run-time state reconstruction

A record has no explicit state, so before the harness predicts a turn the `StateTracker` reads the observed prefix: a
rule whose rendered observation matches the real one is applied deterministically, the remaining turns are explained in
windows of `--state-window` turns by bounded, schema-checked `track_state` model calls, and an eligible rule never
overrides an observed output. With `--official-input` the benchmark's own model input is placed once and verbatim in the
agent's input; with `--knowledge-gate` every worldbook item shown carries an applicability label and foreign values are
masked (`--knowledge-names paths|pages|screens` chooses the identity signal: file paths, web pages, Android screens).
`--environment-prompt-chars` bounds the record's system prompt as the environment description (24,000 by default,
0 for the whole prompt).

## Reading the numbers

One record is one teacher-forced step; there is no rollout. Scores from trajectories that share a task with the
construction traces are not transfer and should be reported separately (`experiments/terminal/awb/construction_task_split.py`,
`experiments/android/row_strata.py`, `experiments/swe/make_folds.py`). Numbers from hash-split or cross-fit trajectories are not leaderboard-comparable.
`REPRODUCING.md` lists the exact flags of the systems reported in the paper.

## Results on the full suite (seven domains)

Trace2Env with GPT-5.6-Sol as the world model backbone, on every record of the seven official test files (2,170
records, 624 trajectories), one run per domain, the official five-dimension judge (gpt-5.2 with the benchmark's per-domain
judge prompts through `awb-judge`), the official aggregation (`awb-score`). The overall score is the unweighted mean over
domains, as in the Qwen-AgentWorld table; the record-weighted mean is 62.01. Intervals are trajectory-clustered 95%
bootstrap intervals of the mean.

| Domain | Records | Trajectories | Worldbook construction data | System | **Trace2Env** | 95% CI |
|---|---|---|---|---|---|---|
| MCP | 286 | 69 | 100 public MCPMark / Toolathlon-Verified trajectories, benchmark-overlapping tasks excluded (`scripts/mcp_sample.py`) | v5.3.1 | **71.59** | [67.25, 75.70] |
| Search | 458 | 93 | the split's own trajectories, task-grouped k=5 cross-fit, 5 worldbooks (`experiments/search/`) | v5.3.1 | **40.24** | [38.30, 42.16] |
| Terminal | 354 | 76 | 20 own Terminus-2 trajectories on Terminal-Bench 2.0 (`experiments/terminal/`) | v5.3.1 | **65.42** | [61.69, 68.96] |
| SWE | 472 | 99 | the split's own trajectories, k=5 cross-fit per scaffold, 10 worldbooks (`experiments/swe/`) | v5.3.1 | **70.59** | [66.43, 74.71] |
| Android | 200 | 92 | the split's own trajectories, k=5 cross-fit per sub-source, 15 worldbooks (`experiments/android/`) | v5.3.3 | **65.10** | [59.97, 70.02] |
| Web | 200 | 118 | 50 own WebArena trajectories on the benchmark's tasks (`experiments/web/`) | v5.3.2 | **59.55** | [55.70, 63.18] |
| OS | 200 | 77 | the split's own trajectories, task-grouped k=5 cross-fit, 5 worldbooks (`experiments/os/`) | v5.3.1 | **71.28** | [65.60, 76.83] |
| **Overall** | 2,170 | 624 | | | **63.40** | |

Judge dimensions (format / factuality / consistency / realism / quality): MCP 90.30 / 58.30 / 74.83 / 67.92 / 66.61;
Search 65.72 / 18.89 / 52.62 / 34.83 / 29.15; Terminal 80.72 / 50.49 / 71.05 / 64.27 / 60.59; SWE 77.12 / 61.94 / 75.00 /
73.04 / 65.87; Android 93.25 / 51.62 / 59.00 / 69.75 / 51.88; Web 93.75 / 43.62 / 51.62 / 65.00 / 43.75; OS 98.25 / 60.38 /
63.00 / 73.38 / 61.38.

Same-backbone baselines (the benchmark's Direct Prompting layout through GPT-5.6-Sol, and Harness only, the runtime with a
schema-only worldbook) were run on five domains: Terminal 55.76 / 60.06, SWE 65.87 / 69.89, Android 62.80 / 62.62, Web
54.23 / 55.40, MCP 72.88 / 71.28. MCP is the one domain where Trace2Env trails the one-shot layout (−1.29 ± 0.91, clustered
CI [−3.04, +0.48]); the gap sits in the agentic harness rather than the worldbook (Harness only − Prompting −1.61 ± 0.62,
worldbook +0.31 ± 0.84), on a split whose official prompt already carries 149k characters of tool definitions and
examples. OS and Search were run with Trace2Env only.

Protocol and hygiene. The benchmark's own model input is placed verbatim in the agent's input (`--official-input`);
state is tracked from the real observed prefix once per trajectory (teacher-forced: the agent's effects never reach a
later record); the knowledge gate labels every worldbook item and withholds format-level items where the official prompt
documents the format; systems v5.3.1 / v5.3.2 (page identity, web) / v5.3.3 (screen identity, android) were defined from
each split's formats before its run. Under the cross-fits a record is predicted with the worldbook built from the
trajectories outside its fold, verified per shard by each environment's `check_crossfit.py`; the Terminal and Web
worldbooks come from our own agent runs on the benchmarks' task suites; the MCP worldbook from public trajectories of
the two suites the split was drawn from, with every trajectory matching a benchmark task excluded (Toolathlon-Verified's
terms permit evaluation use; that worldbook is not redistributed). Cross-fitted and own-trace numbers are not comparable
with each other, and none of the Qwen table's systems uses GPT-5.6-Sol, so the table above mixes backbone and method; the
same-backbone baselines isolate the method. Runs: Terminal 2026-09-22, MCP and Web 2026-09-23, Android and SWE 2026-09-24,
OS 2026-10-15, Search 2026-10-16. The state-path fix of 2026-10-08 (DEVELOPMENT.md) post-dates the first five runs; in
those, 6% (Terminal), 2.6% (SWE), 0.4% (Web), 0% (Android) and 10% (MCP) of the state tracker's mutations were dropped as
unparseable, which can only have left the tracked state less complete. OS and Search GPT calls were billed at OpenRouter's
standard tier and at OpenAI's flex tier respectively (`--chat-provider-json '{"order": ["openai/flex"], "allow_fallbacks":
false}'`, half the standard price at higher latency); the tier changes price and latency, not outputs.
