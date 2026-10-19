# ACP run-history conversion

`acp-export` accepts JSON exports containing `automationId`, `automationName` and `runs`, where
each run has `runId`, `sessionId` and `messages`. Each message has an `index` and a `message`
containing a JSON-RPC object or its serialized JSON string. No model or API key is needed.

```bash
python -m trace2env acp-export /path/to/run-histories.json --output work/acp/onboard
python -m trace2env ingest work/acp/onboard/episodes/*.json --output work/acp/onboard/parsed
python -m trace2env reconstruct work/acp/onboard/episodes/*.json \
  --split-manifest work/acp/onboard/split_manifest.json \
  --environment-id air.onboard --name "YouTrack onboarding" --domain terminal \
  --description @description.md --work-dir work/acp/build --cache-dir work/acp/cache \
  --output work/acp/candidate
```

The output directory must be new. It contains:

- `episodes/`: one file per run with actions, recorded outputs and complete tool lifecycles.
- `incomplete/`: runs with missing outputs, unfinished calls, duplicate IDs, malformed messages,
  wrong-session messages or orphan updates. Their available events and diagnostics remain inspectable.
- `report.json`: counts, per-run paths, skipped/unavailable runs and conversion diagnostics.
- `split_manifest.json`: content digests of the usable episode files and their trajectory groups.

Runs without actions are summarized rather than exported. `--assign-split train|validation|test`
assigns all usable episodes explicitly (default `train`). Sessions are the trajectory groups;
this does not establish independence between separate sessions working on the same task or PR.
Freeze appropriate groups before evaluation. The manifest and report are outside `episodes/`,
so the reconstruction glob and raw-trace corpus do not mistake them for episodes.

## Event mapping

| ACP part | Actor / kind | Content |
|---|---|---|
| `session/prompt` or `user_message_chunk` | `user` / `message` | Recorded prompt |
| `agent_message_chunk` | `agent` / `message` | Recorded text |
| Nonempty `agent_thought_chunk` | `agent` / `message` | Recorded thought; origin labeled in metadata |
| Empty or whitespace-only thought | dropped | — |
| `tool_call` | `agent` / `action` | Uniform normalized action below |
| Result-bearing `tool_call_update` | `environment` / `observation` | Recorded output |
| `plan` | `agent` / `meta` | Plan entries |
| Usage, session info, commands, mode/configuration and unknown updates | episode `control_updates` | Original RPC objects and indexes |
| RPC completions and errors | episode `rpc_messages` | Original RPC objects and indexes |

Adjacent string chunks with the same update type and message ID (or no message ID) concatenate
without inserted separators. Tool/control updates break coalescing. A prompt request and its
immediately echoed identical user message are deduplicated; actual subsequent prompts are retained.
Content with nontext blocks remains structured. No event timestamp is invented from run creation time.

Every tool action is:

```json
{
  "actor": "agent",
  "kind": "action",
  "call_id": "original-toolCallId",
  "content": {
    "type": "execute",
    "arguments": {
      "payload": {
        "title": "original full title",
        "kind": "original ACP category",
        "locations": [],
        "content": []
      }
    }
  }
}
```

Locations, input content (including write text and edit diffs) and title are copied without
tool-specific parsing. If ACP supplies `rawInput`, its value appears in `payload.raw_input`;
`metadata.raw_input_present` distinguishes absence from an explicitly recorded null.
Input-bearing updates can supply it later; source indexes are retained. Missing inputs are not guessed.
Relevant `_meta` (including terminal working directory) is retained in event metadata.

One observation is assembled per call. `rawOutput` takes precedence (strings verbatim, arrays/objects
as JSON values, including empty/null outputs), followed by concatenated
`_meta.terminal_output_delta.data`, then result-bearing update `content`. Repeated `rawOutput`
values replace prior snapshots; only explicit deltas concatenate. Initial tool-call content is
never treated as a result. Lifecycle status and terminal exit information stay in metadata;
`completed` is not interpreted as command success.

The observation occupies the later of the final output and first completion positions in the recorded stream.
Repeated status-only completion notices do not delay an already finished call. Tool IDs
and overlap are preserved: actions and outputs are not rearranged into artificial alternating pairs.
Missing output remains unmatched and makes the run incomplete. Such an earlier call can make later
transitions appear pending to the current segmenter, so incomplete runs are outside the default input glob.

## Reconstruction and simulation interface

The converter produces ordinary raw episode JSON; ingest generates content-derived source and
event IDs. Episode metadata retains the original export path/digest, run/session IDs and indexes.
Missing inputs are an advisory metadata flag, not a new hard reconstruction gate. Correlation and
concurrency use the existing reconstruction checks; ambiguous transitions lose inferred mutations.

The environment description should explain that `execute` means a generic recorded tool call,
with its title and available input fields in `arguments.payload`. Simulation callers must supply
the same representation, including write content or diffs where appropriate. There is no automatic
translation from original native tool names/arguments to this interface. Missing MCP arguments
limit what can be learned; the converter does not claim to recover them.

Raw-trace retrieval includes `payload.title` in action queries. Conversion and ingest checks
establish format compatibility and source fidelity, not next-observation prediction accuracy.
