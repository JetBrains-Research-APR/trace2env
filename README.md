<div align="center">
  <img src="./assets/logo.png" alt="Trace2Env logo" width="200">
</div>

<h1 align="center">
  <b>From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation</b>
</h1>

<h3 align="center">
  <b>Build world model agents that serve as environments for other agents.</b>
</h3>

<div align="center">
  <a href="https://paperswithcode.co/benchmark/agentworldbench?task=world-models&eval=39981">
    <img src="https://paperswithcode.co/api/v1/papers/2610.06100/leaderboard-badge.svg?eval=39981&live=1"
         alt="Papers with Code: SOTA on AgentWorldBench">
  </a>
</div>

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2610.06100"><img src="https://img.shields.io/badge/arXiv-2610.06100-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://huggingface.co/papers/2610.06100"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-page-ffd21e?style=for-the-badge" alt="Hugging Face paper page"></a>
    <a href="https://huggingface.co/spaces/Quanyu001/trace2env_demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-0f8a5f?style=for-the-badge" alt="Live demo"></a>
    <a href="https://quanyulong.net/trace2env/"><img src="https://img.shields.io/badge/Project-page-2457c5?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project page"></a>
    <a href="https://github.com/ruyue0001/trace2env/stargazers"><img src="https://img.shields.io/github/stars/ruyue0001/trace2env?color=f1e05a&style=for-the-badge&logo=star&logoColor=white" alt="Stars"></a>
    <a href="https://github.com/ruyue0001/trace2env/forks"><img src="https://img.shields.io/github/forks/ruyue0001/trace2env?color=2ea44f&style=for-the-badge&logo=git&logoColor=white" alt="Forks"></a>
    <a href="https://github.com/ruyue0001/trace2env/issues"><img src="https://img.shields.io/github/issues/ruyue0001/trace2env?color=d73a49&style=for-the-badge&logo=github&logoColor=white" alt="Issues"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache--2.0-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>

## 🧩 Trace2Env Overview

- **Trace2Env** is a training-free framework for **agentic world modeling**, designed for faithful, stateful, and long-horizon simulation of textual environments.
- Environments built with Trace2Env can be used to train agents without the real system, test agents in safe and reproducible sandboxes, build stateful mocks for tools and APIs, or reconstruct unavailable, private, or legacy environments.
- Trace2Env does not recover the original executable system. Instead, it automatically reconstructs a model of the environment dynamics from observed action–observation trajectories.
- Trace2Env is not tied to a particular environment domain. The [paper](https://arxiv.org/abs/2610.06100) evaluates it on terminals, software repositories, Android and web apps, enterprise services, and text games. The same framework can be applied to other text-interactive systems, including internal enterprise tools, ticketing workflows, cloud and DevOps consoles, API and MCP backends, and other textual worlds.

<div align="center">
  <b>🕹️ A Terminal simulated entirely by a world model agent.</b>
</div>

<div align="center">
  <a href="https://huggingface.co/spaces/Quanyu001/trace2env_demo">
    <img
      src="assets/demo.gif"
      width="800"
      alt="Trace2Env simulating a stateful Terminal environment"
    >
  </a>
</div>

<p align="center">
  <a href="https://huggingface.co/spaces/Quanyu001/trace2env_demo">
    <b>▶ Try the live playground →</b>
  </a>
</p>

<br>

Specifically, Trace2Env works in two phases:

1. **Offline reconstruction.** Past trajectories are processed and organized into a **worldbook** containing environment schemas, transition rules, constraints, invariants, observation contracts, grounded evidence, and demonstrations. Check out what a worldbook looks like [here](https://huggingface.co/datasets/Quanyu001/trace2env-worldbook-terminal).

2. **Agentic simulation.** A **world model agent** actively calls tools to inspect the worldbook, together with the maintained episode state and episodic memory, and infers the next observation together with the state changes that should persist. A shared runtime harness verifies the proposal against the schemas, rules, and invariants and commits the accepted changes, so the consequences of a task agent's action carry into later turns.

<div align="center">
  <img src="./assets/overview.png" width="1000" alt="Trace2Env framework overview">
</div>


### ✨ Highlights

- **Build from tens of traces.** Worldbook construction needs only recorded interactions, not access to the original executable environment, its source code, dependencies, or infrastructure.

- **Agentic world modeling.** The world model agent actively consults the knowledge needed for each action instead of relying on a single flattened environment prompt.

- **Stateful, long-horizon simulation.** Verified state changes persist across turns, allowing the simulated environment to preserve the consequences of earlier actions throughout an interaction.

- **Evaluated as a world model.** Trace2Env supports both next-observation prediction ([Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld)) and successive interaction with the world model to evaluate its long-horizon consistency.

## 📰 News

- 🏆 **[2026-10]** Trace2Env (gpt-5.6-sol) ranks first on the [AgentWorldBench leaderboard](https://paperswithcode.co/benchmark/agentworldbench?task=world-models&eval=39981)
  on Papers with Code with an overall score of **63.40** across all seven domains (MCP, Search, Terminal, SWE, Android, Web, OS), ahead of Qwen-AgentWorld-397B-A17B (58.71). 
- 🚀 **[2026-10]** Trace2Env is released. The paper, *From Traces to Agentic Worlds: Agentic Language World Models for
  Interactive Environment Simulation*, is on [arXiv](https://arxiv.org/abs/2610.06100) and
  [Hugging Face Papers](https://huggingface.co/papers/2610.06100); the [project page](https://quanyulong.net/trace2env/)
  summarizes the method and results, and the [live demo](https://huggingface.co/spaces/Quanyu001/trace2env_demo) lets you
  play against a reconstructed terminal.

## Install and try it

Python 3.11 or newer. The demo below needs no model and no API key.

```bash
git clone --recurse-submodules https://github.com/ruyue0001/trace2env.git
cd trace2env
python -m pip install -e '.[dev]'
python -m pytest                                   # 287 tests, no network

python -m trace2env init-demo --output .trace2env/demo-authored
python -m trace2env inspect .trace2env/demo-authored --action withdraw
python -m trace2env replay .trace2env/demo-authored \
  --cases examples/replay/ledger_cases.jsonl --output work/demo-replay.json \
  --promote-to .trace2env/demo-validated
python -m trace2env simulate .trace2env/demo-validated --session .trace2env/demo-session \
  --action '{"type":"withdraw","arguments":{"amount":25}}'     # -> "Withdrew 25. Balance: 75.0"
```

The demo worldbook is hand-authored to show the machinery: replay validation against reference cases, promotion to a
validated package, and a persistent session. In the code, a worldbook is called a *package* and one episode's state and
audit a *session*. [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) follows it file by file.
Package destinations (`--output`, `--promote-to`) must not already exist.

To skip the build and try the world model on a real reconstructed environment, download the paper's terminal
worldbook from Hugging Face (20 Terminal-Bench 2.0 traces; the one the live demo runs on) and use it as `WORLDBOOK`
in the commands below:

```bash
pip install huggingface_hub
hf download Quanyu001/trace2env-worldbook-terminal --repo-type dataset --local-dir work/worldbooks/terminal
python -m trace2env inspect work/worldbooks/terminal/worldbook --action cat     # WORLDBOOK=work/worldbooks/terminal/worldbook
```

## Build a worldbook from your traces

A trace records one episode: the actions an agent took and the observations the environment returned. The simplest
input is JSONL with one event per line; JSON and role-prefixed text are accepted too, and converters for Terminal-Bench,
WebArena, AgentWorldBench, EnvScaler, ALFWorld and SciWorld traces are in
[docs/TRACE_COLLECTION.md](docs/TRACE_COLLECTION.md). Tens of episodes are enough to start.

```jsonl
{"episode_id": "ep1", "actor": "agent", "kind": "action", "content": {"type": "shell", "arguments": {"commands": ["ls"]}}}
{"episode_id": "ep1", "actor": "tool", "kind": "observation", "content": "report.txt  notes.md"}
```

Write a short `description.md` saying what the environment is and what its actions and observations look like, point
the model options at an available gateway model, then build. Configure `LITELLM_API_BASE` and `LITELLM_API_KEY`
in the repository-root `.env` first (see `.env.example`); the default model is `openai/gpt-6.1-sol`.

```bash
MODEL=(--provider chat --model openai/gpt-6.1-sol)
python -m trace2env ingest TRACES --output work/parsed                  # deterministic parse and segmentation, no model
python -m trace2env reconstruct TRACES --environment-id my.env --name "My environment" \
  --description @description.md --domain terminal "${MODEL[@]}" \
  --work-dir work/build --cache-dir work/cache --output work/my-env-v1
python -m trace2env inspect work/my-env-v1 --action some_action         # what the worldbook knows about one action
```

A fresh worldbook is a *candidate* until replay validation promotes it ([docs/OFFLINE_VALIDATION.md](docs/OFFLINE_VALIDATION.md)),
which is why the commands below pass `--allow-unvalidated`. Models that do not return reliable JSON can run the agent
with native tool calling (`--agent-transport tools`). For scale: the paper's terminal worldbook, from 20 traces, cost
about $27 and an hour to build, and a simulated step costs a few tens of cents with a large hosted model.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) says what each generated file is; for research runs, `scripts/pilot_stages.py`
runs the reconstruction stage by stage and keeps an inspectable version of every stage.

## Use it as a world model

[docs/RUNTIME_HARNESS.md](docs/RUNTIME_HARNESS.md) explains how one action is simulated: rule selection, the agent's
brief and tools, verification, trust policy, rendering, commit. Two ways to use a worldbook are built in.

**Successive interaction.** One persistent session, one call per task-agent action; state and episodic memory carry
over, so later observations reflect earlier actions. Actions follow the worldbook's action schema
(`trace2env inspect WORLDBOOK --action NAME` shows it); the terminal worldbook's `shell` action takes a list of commands.

```bash
# MODEL as defined above; the session keeps state between calls
python -m trace2env simulate WORLDBOOK --session work/session-1 --allow-unvalidated --agent "${MODEL[@]}" \
  --initial-state '{"session": {"cwd": "/app"}}' --action '{"type": "shell", "arguments": {"commands": ["echo hi > report.txt"]}}'
python -m trace2env simulate WORLDBOOK --session work/session-1 --agent "${MODEL[@]}" \
  --action '{"type": "shell", "arguments": {"commands": ["mv report.txt summary.txt"]}}'
python -m trace2env simulate WORLDBOOK --session work/session-1 --agent "${MODEL[@]}" \
  --action '{"type": "shell", "arguments": {"commands": ["cat report.txt"]}}'      # -> cat: report.txt: No such file or directory
```

A programmatic task agent drives the same loop through `RuntimeHarness.step`; the closed-loop runners wrap it with an
LLM task agent and replay its actions in the real environment: `envscaler-long-horizon` for EnvScaler environments
([docs/LONG_HORIZON.md](docs/LONG_HORIZON.md)) and the ALFWorld and SciWorld scripts
([docs/ALFWORLD_TRACE2ENV.md](docs/ALFWORLD_TRACE2ENV.md)).

**Next-state prediction on [AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench).** Predict every
record of a split with the harness, judge with the official prompts, score with the official aggregation
([docs/AGENTWORLDBENCH.md](docs/AGENTWORLDBENCH.md)); `--mode prompting` runs the official Direct Prompting baseline instead.

```bash
python -m trace2env awb-run work/agentworldbench/terminal_test.jsonl --mode agentic --package WORLDBOOK \
  --official-input --knowledge-gate --features default,evidence --allow-unvalidated --output work/awb/pred.jsonl \
  "${MODEL[@]}" --cache-dir work/awb/cache
python -m trace2env awb-judge --predictions work/awb/pred.jsonl --output work/awb/judged.jsonl \
  --judge-model openai/gpt-5.2-2025-12-11 \
  --cache-dir work/awb/judge-cache                                       # the benchmark's official judge is gpt-5.2
python -m trace2env awb-score --predictions work/awb/judged.jsonl --summary work/awb/score.json
```

All Trace2Env model requests use the internal LiteLLM gateway, configured by `LITELLM_API_BASE` and
`LITELLM_API_KEY` in the repository-root `.env`. Environment variables with these names override the file.
Use the gateway's model IDs (default `openai/gpt-6.1-sol`). This applies to reconstruction, runtime
agents, task agents, collection scripts, and judges. Missing settings fail before sending a request.
`--base-url`, `--api-key-env`, and their judge equivalents have been removed. The Python environment
running the standalone scripts must have this project installed (`pip install -e .`). For external
benchmark runners, `python -m trace2env.llm_config run COMMAND ...` supplies the same configuration.
DeepSeek experiment launchers require `DEEPSEEK_MODEL` to name an available gateway model.
Non-judge temperatures default to `1.0`; judge defaults remain at `0.6`.
See [LiteLLM configuration and usage](docs/LITELLM.md) for model and sampling overrides.
The frozen third-party `baseline/Word2World` submodule is unchanged; these settings apply to
Trace2Env's entry points and the external harnesses launched by its scripts.

## 📊 Results and reproduction

Trace2Env with GPT-5.6-Sol as the world model backbone, one run per domain on every record of the official test files,
with the official gpt-5.2 judge; the overall score is the unweighted mean over domains, as in the benchmark's own table.

| | MCP | Search | Terminal | SWE | Android | Web | OS | **Overall** |
|---|---|---|---|---|---|---|---|---|
| Trace2Env (gpt-5.6-sol) | 71.59 | 40.24 | 65.42 | 70.59 | 65.10 | 59.55 | 71.28 | **63.40** |
| Qwen-AgentWorld-397B-A17B (their table) | 68.24 | 37.82 | 57.73 | 68.49 | 60.20 | 50.98 | 67.89 | 58.71 |

Where each worldbook's construction data came from, the same-backbone baselines, intervals, audits and caveats are in
[docs/AGENTWORLDBENCH.md](docs/AGENTWORLDBENCH.md).

**Reproducing the paper.** [REPRODUCING.md](REPRODUCING.md) maps every experiment of the paper to its data, scripts and flags. The scripts of each
environment are under [`experiments/`](experiments/). The frozen Terminal worldbook is published on
[Hugging Face](https://huggingface.co/datasets/Quanyu001/trace2env-worldbook-terminal) and also runs the live demo; the other frozen
worldbooks and the collected traces are available on request.

## Repository layout

| Path | Contents                                                                                                                                                                 |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/trace2env/` | Library and CLI: trace adapters, reconstruction, compiler, runtime harness, world model agent, state tracking, benchmark adapters, long-horizon runner                   |
| `tests/` | 287 tests with scripted models; no network                                                                                                                               |
| `scripts/` | Stage-by-stage reconstruction driver, trace collection (Terminal-Bench 2.0 with Harbor, WebArena through Playwright MCP), ALFWorld and SciWorld runs, worldbook ablation |
| `experiments/` | Launchers, cross-fit tooling, audits and report scripts of the paper's experiments, per environment                                                                      |
| `docs/` | Architecture, runtime harness, offline validation, prompt contracts, trace collection, benchmark protocol, long-horizon evaluation                                       |
| `examples/` | The authored ledger demo and a synthetic benchmark file for smoke tests                                                                                                  |
| `baseline/Word2World` | Submodule used by the ALFWorld and SciWorld scripts (task lists, grammar, simulator adapters)                                                                            |

[DEVELOPMENT.md](DEVELOPMENT.md) has the commands, the code map and the design constraints to preserve.


## 📚 Citation

If you find Trace2Env useful in your research, please consider citing our paper:

```bibtex
@article{long2026agenticworlds,
  title={From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation},
  author={Long, Quanyu and Chen, Xiao and Chen, Jianda and Zhang, Haozhen and Hu, Qisheng and Bao, Jianzhu and Wang, Wenya},
  journal={arXiv preprint arXiv:2610.06100},
  year={2026}
}
```
