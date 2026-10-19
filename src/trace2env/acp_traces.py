"""ACP run-history exports → ordered Trace2Env episodes, without model calls.

Every tool has the same interface: ``execute(payload={title, kind, locations, content})``.
The payload copies recorded inputs, including optional rawInput; it never guesses missing
arguments or folds returned output into an action. Original exports remain the source of truth.
Incomplete runs are retained separately so a glob of ``episodes/*.json`` is reconstruction-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from trace2env.adapters import stable_id
from trace2env.models import SplitAssignment, SplitManifest
from trace2env.storage import write_json

VERSION = "acp-v1"
TERMINAL_STATUSES = {"completed", "failed"}
MESSAGE_UPDATES = {
    "user_message_chunk": "user",
    "agent_message_chunk": "agent",
    "agent_thought_chunk": "agent",
}
CONTROL_UPDATES = {
    "usage_update", "session_info_update", "available_commands_update",
    "current_mode_update", "config_option_update",
}
_MISSING = object()


def _content(value: Any) -> Any:
    """Text blocks concatenate without invented separators; nontext stays structured."""
    if isinstance(value, dict) and value.get("type") == "text" and isinstance(value.get("text"), str):
        return value["text"]
    if isinstance(value, list) and value and all(
        isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        for part in value
    ):
        return "".join(part["text"] for part in value)
    return value


def _result_content(parts: Any) -> Any:
    if not parts:
        return _MISSING
    if isinstance(parts, list):
        parts = [part["content"] if isinstance(part, dict) and part.get("type") == "content"
                 and "content" in part else part for part in parts]
    return _content(parts)


@dataclass
class _Call:
    action: dict[str, Any]
    status: str = "pending"
    indexes: list[Any] = field(default_factory=list)
    raw_output: Any = _MISSING
    result_content: Any = _MISSING
    deltas: list[str] = field(default_factory=list)
    output_position: int = -1
    terminal_position: int = -1
    meta: dict[str, Any] = field(default_factory=dict)

    def update(self, update: dict[str, Any], index: Any, position: int, *, initial: bool = False) -> None:
        self.indexes.append(index)
        if "status" in update:
            self.status = update["status"]
            if self.status in TERMINAL_STATUSES and self.terminal_position < 0:
                # Repeated completion notices must not turn an already finished call into an overlap.
                self.terminal_position = position
        if "rawInput" in update:
            self.action["content"]["arguments"]["payload"]["raw_input"] = update["rawInput"]
            self.action["metadata"]["raw_input_present"] = True
            self.action["metadata"].setdefault("input_source_indexes", []).append(index)
        if "rawOutput" in update:
            self.raw_output = update["rawOutput"]  # replacements, not deltas; empty/null are real values
            self.output_position = position
        meta = update.get("_meta") or {}
        self.meta.update({key: value for key, value in meta.items() if key != "terminal_output_delta"})
        delta = meta.get("terminal_output_delta") or {}
        if isinstance(delta, dict) and isinstance(delta.get("data"), str):
            self.deltas.append(delta["data"])
            self.output_position = position
        # Initial content describes the action (e.g. a write), never its result.
        if not initial and "content" in update:
            content = _result_content(update["content"])
            if content is not _MISSING:
                self.result_content = content
                self.output_position = position

    def output(self) -> tuple[Any, str]:
        if self.raw_output is not _MISSING:
            return self.raw_output, "rawOutput"
        if self.deltas:
            return "".join(self.deltas), "terminal_output_delta"
        return self.result_content, "content"


def run_episode(run: dict[str, Any], export: dict[str, Any], *, source_path: str = "",
                source_sha256: str = "", split: str = "train") -> dict[str, Any]:
    """Convert one run. Warnings retain source indexes; missing results remain unmatched."""
    run_id = str(run["runId"])
    events: list[tuple[int, dict[str, Any]]] = []
    calls: dict[str, _Call] = {}
    diagnostics: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    rpc: list[dict[str, Any]] = []
    last_text: dict[str, Any] | None = None
    dropped_thoughts = 0

    def diagnose(code: str, *, incomplete: bool = True, **details: Any) -> None:
        diagnostics.append({"code": code, "incomplete": incomplete, **details})

    def emit(position: int, actor: str, kind: str, content: Any, index: Any,
             tag: str, *, call_id: str | None = None, **metadata: Any) -> dict[str, Any]:
        event = {"id": stable_id("acp_evt", run_id, position, tag, call_id),
                 "actor": actor, "kind": kind, "content": content,
                 "metadata": {"acp_update": tag, "source_indexes": [index], **metadata}}
        if call_id is not None:
            event["call_id"] = call_id
        events.append((position, event))
        return event

    previous_index: int | None = None
    for position, part in enumerate(run.get("messages") or []):
        index = part.get("index", position) if isinstance(part, dict) else position
        if isinstance(index, int):
            if previous_index is not None and index <= previous_index:
                diagnose("non_increasing_index", source_index=index)
            previous_index = index
        try:
            message = part["message"]
            message = json.loads(message) if isinstance(message, str) else message
            if not isinstance(message, dict):
                raise ValueError("RPC message must be an object")
            params = message.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("RPC params must be an object")
            update = params.get("update") or {}
            if not isinstance(update, dict):
                raise ValueError("ACP update must be an object")
            if "sessionUpdate" in update and not isinstance(update["sessionUpdate"], str):
                raise ValueError("sessionUpdate must be a string")
            if any(obj.get("_meta") is not None and not isinstance(obj["_meta"], dict) for obj in (params, update)):
                raise ValueError("RPC/ACP metadata must be an object")
        except (KeyError, TypeError, ValueError) as exc:
            last_text = None
            diagnose("malformed_message", source_index=index, detail=str(exc))
            continue
        session = params.get("sessionId")
        if session and run.get("sessionId") and session != run["sessionId"]:
            last_text = None
            diagnose("session_mismatch", source_index=index, session_id=session)
            controls.append({"source_index": index, "message": message})
            continue
        method = message.get("method")
        tag = update.get("sessionUpdate") if method == "session/update" else None
        if tag in {"tool_call", "tool_call_update"}:
            # Invalid optional fields must not crash the entire export or silently become inputs.
            invalid = [key for key, expected in (("status", str), ("_meta", dict), ("content", list), ("locations", list))
                       if key in update and update[key] is not None and not isinstance(update[key], expected)]
            if invalid:
                last_text = None
                diagnose("malformed_tool_update", source_index=index, fields=invalid)
                controls.append({"source_index": index, "message": message})
                continue
        if method == "session/prompt" or tag in MESSAGE_UPDATES:
            tag = tag or "session/prompt"
            actor = MESSAGE_UPDATES.get(tag, "user")
            content = _content(params.get("prompt", []) if tag == "session/prompt" else update.get("content"))
            if tag == "agent_thought_chunk" and isinstance(content, str) and not content.strip():
                dropped_thoughts += 1
                continue
            message_id = update.get("messageId")
            if (tag != "session/prompt" and last_text is not None and isinstance(content, str)
                    and isinstance(last_text["content"], str)
                    and last_text["metadata"]["acp_update"] == tag
                    and last_text["metadata"].get("message_id") == message_id):
                last_text["content"] += content
                last_text["metadata"]["source_indexes"].append(index)
            else:
                last_text = emit(position, actor, "message", content, index, tag, message_id=message_id,
                                 acp_meta={**(params.get("_meta") or {}), **(update.get("_meta") or {})})
            continue
        last_text = None  # never coalesce text across a tool or control update
        if tag == "tool_call":
            call_id = update.get("toolCallId")
            if not isinstance(call_id, str) or not call_id:
                diagnose("missing_call_id", source_index=index)
                controls.append({"source_index": index, "message": message})
                continue
            payload = {"title": update.get("title", ""), "kind": update.get("kind", "other"),
                       "locations": update.get("locations") or [], "content": update.get("content") or []}
            action = emit(position, "agent", "action", {"type": "execute", "arguments": {"payload": payload}},
                          index, tag, call_id=call_id, acp_status=update.get("status", "pending"),
                          raw_input_present=False, acp_meta=update.get("_meta") or {})
            if call_id in calls:
                diagnose("duplicate_call_id", call_id=call_id, source_index=index)
                continue
            call = calls[call_id] = _Call(action)
            call.update(update, index, position, initial=True)
        elif tag == "tool_call_update":
            call_id = update.get("toolCallId")
            if not isinstance(call_id, str) or call_id not in calls:
                diagnose("orphan_tool_update", call_id=call_id, source_index=index)
                controls.append({"source_index": index, "message": message})
                continue
            calls[call_id].update(update, index, position)
        elif tag == "plan":
            emit(position, "agent", "meta", update.get("entries", []), index, tag)
        elif method == "session/update":
            controls.append({"source_index": index, "message": message})
            if tag not in CONTROL_UPDATES:
                diagnose("unknown_update", incomplete=False, source_index=index, session_update=tag)
        else:
            rpc.append({"source_index": index, "message": message})
            if "error" in message:
                diagnose("rpc_error", incomplete=False, source_index=index, error=message["error"])

    for call_id, call in calls.items():
        output, output_source = call.output()
        if call.status not in TERMINAL_STATUSES:
            diagnose("nonterminal_call", call_id=call_id, status=call.status)
        if output is _MISSING:
            diagnose("missing_output", call_id=call_id, status=call.status)
            continue
        position = max(call.output_position, call.terminal_position)
        event = emit(position, "environment", "observation", output, call.indexes[-1], "tool_call_update",
                     call_id=call_id, acp_status=call.status, output_source=output_source, acp_meta=call.meta)
        event["metadata"]["source_indexes"] = call.indexes

    # A request and its immediately echoed user message are one prompt, not two turns.
    ordered: list[dict[str, Any]] = []
    for _, event in sorted(events, key=lambda pair: pair[0]):
        previous = ordered[-1] if ordered else None
        if (previous is not None and previous["actor"] == event["actor"] == "user"
                and previous["content"] == event["content"]
                and {previous["metadata"]["acp_update"], event["metadata"]["acp_update"]}
                == {"session/prompt", "user_message_chunk"}):
            previous["metadata"]["source_indexes"].extend(event["metadata"]["source_indexes"])
            previous["metadata"]["prompt_echo_deduplicated"] = True
        else:
            ordered.append(event)
    return {
        "episode_id": f"acp:{run_id}", "source": "acp", "converter_version": VERSION,
        "automation_id": export.get("automationId"), "automation_name": export.get("automationName"),
        "session_id": run.get("sessionId"), "run_status": run.get("status"),
        "created_at": run.get("createdAt"), "trace_id": run.get("traceId"),
        "trajectory_group": f"acp:{run.get('sessionId') or run_id}", "split": split,
        "source_export": {"path": source_path, "sha256": source_sha256, "run_id": run_id},
        "incomplete": any(item["incomplete"] for item in diagnostics),
        "diagnostics": diagnostics, "control_updates": controls, "rpc_messages": rpc,
        "dropped_empty_thoughts": dropped_thoughts, "events": ordered,
    }


def export_histories(inputs: Iterable[str | Path], output_dir: str | Path, *, split: str = "train"
                     ) -> tuple[list[Path], SplitManifest, dict[str, Any]]:
    """Export into a fresh directory: usable episodes/, incomplete/, manifest and report.

    No-activity runs are summarized rather than exported; source histories remain untouched.
    Split assignment is explicit (no assumed random independence between related workflows).
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    paths: list[Path] = []
    assignments: dict[str, SplitAssignment] = {}
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for input_path in inputs:
        path = Path(input_path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        export = json.loads(raw)
        if not isinstance(export, dict) or not isinstance(export.get("runs"), list):
            raise ValueError(f"Not an ACP run-history export: {path}")
        for run in export["runs"]:
            key = (export.get("automationId"), str(run["runId"]))
            if key in seen:
                raise ValueError(f"Duplicate automation/run identity: {key}")
            seen.add(key)
            episode = run_episode(run, export, source_path=str(path.resolve()), source_sha256=digest, split=split)
            calls = sum(event["kind"] == "action" for event in episode["events"])
            observations = sum(event["kind"] == "observation" for event in episode["events"])
            summary = {"run_id": run["runId"], "automation_id": export.get("automationId"),
                       "source_export": episode["source_export"],
                       "status": run.get("status"), "calls": calls, "observations": observations,
                       "incomplete": episode["incomplete"], "diagnostics": episode["diagnostics"],
                       "raw_inputs_missing": sum(event["kind"] == "action" and not event["metadata"]["raw_input_present"]
                                                 for event in episode["events"]),
                       "dropped_empty_thoughts": episode["dropped_empty_thoughts"]}
            if not calls:
                summary["skipped"] = "no environment actions"
                summary["unavailable_reason"] = run.get("unavailableReason")
            else:
                target = output / ("incomplete" if episode["incomplete"] else "episodes") / f"{stable_id('acp', *key)}.json"
                write_json(target, episode)
                summary["path"] = str(target.relative_to(output))
                if not episode["incomplete"]:
                    paths.append(target)
                    source_id = "src_" + hashlib.sha256(target.read_bytes()).hexdigest()
                    assignments[source_id] = SplitAssignment(split=split, trajectory_group=episode["trajectory_group"])
            summaries.append(summary)
    report = {"converter_version": VERSION, "runs": len(summaries), "episodes": len(paths),
              "incomplete_runs": sum(item["incomplete"] and "path" in item for item in summaries),
              "skipped_runs": sum("skipped" in item for item in summaries),
              "calls": sum(item["calls"] for item in summaries),
              "observations": sum(item["observations"] for item in summaries), "run_summaries": summaries}
    manifest = SplitManifest(assignments=assignments)
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "report.json", report)
    return paths, manifest, report
