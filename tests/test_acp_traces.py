"""ACP conversion contracts. All fixtures are synthetic; no private data or live model."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from trace2env.acp_traces import export_histories, run_episode
from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.cli import main
from trace2env.compiler import EnvironmentCompiler
from trace2env.llm import ScriptedLLM
from trace2env.models import (
    ActionSchema, ActionSpec, ArgumentSpec, Condition, Confidence, LocalTransitionEvidence,
    NormalizedAction, Operand, ReconstructionArtifacts, ReconstructionConfig, RenderContract,
    StateField, StateMutation, StateSchema, StepRequest, TransitionRule,
)
from trace2env.reconstruction import TraceReconstructor
from trace2env.runtime import RuntimeHarness
from trace2env.storage import read_json, write_json
from trace2env.trace_corpus import TraceCorpus, action_query


def update(tag: str, **fields):
    return {"jsonrpc": "2.0", "method": "session/update",
            "params": {"sessionId": "session-1", "update": {"sessionUpdate": tag, **fields}}}


def text(tag: str, value: str, **fields):
    return update(tag, content={"type": "text", "text": value}, **fields)


def call(call_id="a", **fields):
    return update("tool_call", toolCallId=call_id, title="git status", kind="execute",
                  status="in_progress", content=[], locations=[], **fields)


def result(call_id="a", **fields):
    return update("tool_call_update", toolCallId=call_id, status="completed", **fields)


def history(messages, run_id="run-1"):
    return {"runId": run_id, "sessionId": "session-1", "status": "FINISHED",
            "createdAt": "2026-09-01T10:00:00Z", "traceId": "trace-1",
            "messages": [{"index": i, "message": json.dumps(message)} for i, message in enumerate(messages)]}


EXPORT = {"automationId": "automation-1", "automationName": "Onboard"}


class AcpTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def convert(self, messages):
        return run_episode(history(messages), EXPORT)

    def load(self, episode):
        path = self.root / "episode.json"
        write_json(path, episode)
        return load_raw_traces(path)[0]

    def test_prompt_chunks_echo_thoughts_and_nontext(self):
        prompt = {"method": "session/prompt", "params": {"sessionId": "session-1",
                  "prompt": [{"type": "text", "text": "Hello"}]}}
        episode = self.convert([
            prompt, text("user_message_chunk", "Hel"), text("user_message_chunk", "lo"),
            text("agent_thought_chunk", ""), text("agent_thought_chunk", " \n"),
            text("agent_thought_chunk", "Think", messageId="thought"),
            text("agent_thought_chunk", "ing", messageId="thought"),
            text("agent_message_chunk", "Next", messageId="m1"),
            text("agent_message_chunk", " step", messageId="m1"),
            text("agent_message_chunk", "New message", messageId="m2"),
            update("agent_message_chunk", content={"type": "image", "data": "abc"}),
            call(), result(rawOutput="clean"), prompt,
        ])
        events = episode["events"]
        self.assertEqual([e["content"] for e in events[:4]], ["Hello", "Thinking", "Next step", "New message"])
        self.assertEqual(events[0]["metadata"]["source_indexes"], [0, 1, 2])
        self.assertTrue(events[0]["metadata"]["prompt_echo_deduplicated"])
        self.assertEqual(events[1]["metadata"]["acp_update"], "agent_thought_chunk")
        self.assertEqual(events[4]["content"], {"type": "image", "data": "abc"})
        self.assertEqual(events[-1]["content"], "Hello")  # later real user turn retained
        self.assertEqual(episode["dropped_empty_thoughts"], 2)
        self.assertFalse(any("timestamp" in event for event in events))

    def test_input_content_and_diff_are_not_responses(self):
        for content in ([{"type": "content", "content": {"type": "text", "text": "new file"}}],
                        [{"type": "diff", "path": "/x", "oldText": "before", "newText": "after"}]):
            initial = update("tool_call", toolCallId="a", title="Write x", kind="edit", status="in_progress",
                             content=content, locations=[{"path": "/x"}], rawInput=None,
                             _meta={"terminal_info": {"cwd": "/workspace"}})
            episode = self.convert([initial, result(rawOutput="written")])
            action, observation = episode["events"]
            payload = action["content"]["arguments"]["payload"]
            self.assertEqual(payload, {"title": "Write x", "kind": "edit", "content": content,
                                      "locations": [{"path": "/x"}], "raw_input": None})
            self.assertEqual(action["content"]["type"], "execute")
            self.assertTrue(action["metadata"]["raw_input_present"])
            self.assertEqual(action["metadata"]["acp_meta"]["terminal_info"]["cwd"], "/workspace")
            self.assertEqual(observation["content"], "written")
            missing = self.convert([initial])
            self.assertEqual(len(missing["events"]), 1)
            self.assertTrue(missing["incomplete"])

    def test_lifecycle_snapshots_and_order_preserve_concurrency(self):
        episode = self.convert([
            call("a"), call("b"),
            update("tool_call_update", toolCallId="a", status="in_progress", rawOutput="partial"),
            result("b", rawOutput="B"), result("a", rawOutput="A"),
            result("a"),  # status-only completion must not erase the final output
        ])
        self.assertEqual([(e["kind"], e["call_id"]) for e in episode["events"]],
                         [("action", "a"), ("action", "b"), ("observation", "b"), ("observation", "a")])
        self.assertEqual(episode["events"][-1]["content"], "A")
        loaded = self.load(episode)
        slices = segment_transitions(loaded)
        self.assertEqual(len(slices), 2)
        self.assertTrue(all(s.alignment == "correlated" and len(s.observation_event_ids) == 1 for s in slices))
        self.assertTrue(all(s.concurrent_action_event_ids for s in slices))
        by_id = {e.id: e for e in loaded.events}
        self.assertEqual([by_id[s.observation_event_ids[0]].content for s in slices], ["A", "B"])

    def test_terminal_deltas_status_and_late_raw_input(self):
        episode = self.convert([
            call(_meta={"terminal_info": {"cwd": "/workspace"}}),
            update("tool_call_update", toolCallId="a", rawInput={"command": "git status"},
                   _meta={"terminal_output_delta": {"data": "fi"}}),
            result(_meta={"terminal_output_delta": {"data": "nal"}, "terminal_exit": {"exit_code": 1}}),
        ])
        action, observation = episode["events"]
        self.assertEqual(action["content"]["arguments"]["payload"]["raw_input"], {"command": "git status"})
        self.assertEqual(action["metadata"]["input_source_indexes"], [1])
        self.assertEqual(observation["content"], "final")
        self.assertEqual(observation["metadata"]["output_source"], "terminal_output_delta")
        self.assertEqual(observation["metadata"]["acp_meta"],
                         {"terminal_info": {"cwd": "/workspace"}, "terminal_exit": {"exit_code": 1}})
        self.assertEqual(observation["metadata"]["acp_status"], "completed")  # not inferred success

    def test_repeated_completion_does_not_create_false_overlap(self):
        episode = self.convert([call("a"), result("a", rawOutput="A"),
                                call("b"), result("a"), result("b", rawOutput="B")])
        self.assertEqual([(e["kind"], e["call_id"]) for e in episode["events"]],
                         [("action", "a"), ("observation", "a"), ("action", "b"), ("observation", "b")])
        self.assertFalse(any(s.concurrent_action_event_ids for s in segment_transitions(self.load(episode))))

    def test_raw_outputs_including_empty_null_and_arrays_are_preserved(self):
        for output in ("", None, [], {"answer": 1}, [{"type": "text", "text": "MCP text"}]):
            episode = self.convert([call(), result(rawOutput=output, content=[{"type": "content", "content":
                                   {"type": "text", "text": "fallback"}}],
                                   _meta={"terminal_output_delta": {"data": "delta"}})])
            self.assertEqual(episode["events"][-1]["content"], output)
            self.assertFalse(episode["incomplete"])

    def test_update_content_fallback_and_failed_output(self):
        parts = [{"type": "content", "content": {"type": "text", "text": "fi"}},
                 {"type": "content", "content": {"type": "text", "text": "nal"}}]
        episode = self.convert([call(), update("tool_call_update", toolCallId="a", status="failed", content=parts)])
        event = episode["events"][-1]
        self.assertEqual((event["actor"], event["kind"], event["content"]), ("environment", "observation", "final"))
        self.assertEqual(event["metadata"]["acp_status"], "failed")
        self.assertFalse(episode["incomplete"])

    def test_control_updates_do_not_fill_history_and_rpc_errors_are_not_tool_results(self):
        episode = self.convert([
            call(), update("usage_update", used=123, size=456),
            update("session_info_update", title="Session"), update("future_update", value=1),
            update("plan", entries=[{"content": "Do task"}]),
            {"id": 0, "result": {"stopReason": "end_turn", "_meta": {"start_revision": "abc"}}},
            {"id": 1, "error": {"code": -1, "message": "transport failed"}}, result(rawOutput="ok"),
        ])
        self.assertEqual([e["kind"] for e in episode["events"]], ["action", "meta", "observation"])
        self.assertEqual(len(episode["control_updates"]), 3)
        self.assertEqual(len(episode["rpc_messages"]), 2)
        self.assertEqual({d["code"] for d in episode["diagnostics"]}, {"unknown_update", "rpc_error"})
        self.assertFalse(episode["incomplete"])

    def test_tool_updates_break_text_coalescing(self):
        episode = self.convert([call(), text("agent_message_chunk", "before"),
                                result(rawOutput="ok"), text("agent_message_chunk", "after")])
        self.assertEqual([e["content"] for e in episode["events"] if e["kind"] == "message"], ["before", "after"])

    def test_adjacent_prompt_requests_are_separate_turns(self):
        prompts = [{"method": "session/prompt", "params": {"prompt": [{"type": "text", "text": value}],
                   "_meta": {"modelId": "model"}}} for value in ("first", "second")]
        episode = self.convert(prompts)
        self.assertEqual([e["content"] for e in episode["events"]], ["first", "second"])
        self.assertEqual(episode["events"][0]["metadata"]["acp_meta"], {"modelId": "model"})

    def test_incomplete_and_malformed_runs_are_diagnosed(self):
        cases = [([call()], "missing_output"),
                 ([result(rawOutput="orphan")], "orphan_tool_update"),
                 ([call(), call(), result(rawOutput="x")], "duplicate_call_id"),
                 ([call(), update("tool_call_update", toolCallId="a", rawOutput="partial")], "nonterminal_call"),
                 ([call(), update("tool_call_update", toolCallId="a", status=[])], "malformed_tool_update"),
                 ([update("tool_call_update", toolCallId=[])], "orphan_tool_update"),
                 ([update("tool_call", title="no ID")], "missing_call_id")]
        for messages, diagnostic in cases:
            episode = self.convert(messages)
            self.assertTrue(episode["incomplete"])
            self.assertIn(diagnostic, {d["code"] for d in episode["diagnostics"]})
        run = history([call(), result(rawOutput="ok")])
        run["messages"].insert(0, {"index": -1, "message": "{invalid"})
        self.assertIn("malformed_message", {d["code"] for d in run_episode(run, EXPORT)["diagnostics"]})
        wrong = call()
        wrong["params"]["sessionId"] = "other"
        self.assertTrue(self.convert([wrong])["incomplete"])
        run = history([call(), result(rawOutput="ok")])
        run["messages"][1]["index"] = 0
        self.assertIn("non_increasing_index", {d["code"] for d in run_episode(run, EXPORT)["diagnostics"]})

    def test_export_cli_separates_incomplete_and_empty_runs_and_binds_digests(self):
        source = self.root / "histories.json"
        write_json(source, {**EXPORT, "runs": [history([call(), result(rawOutput="ok")]),
                                               history([call()], "unfinished"), history([], "empty")]})
        output = self.root / "export"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(["acp-export", str(source), "--output", str(output), "--assign-split", "validation"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["episodes"], 1)
        report = read_json(output / "report.json")
        self.assertEqual((report["runs"], report["incomplete_runs"], report["skipped_runs"]), (3, 1, 1))
        paths = list((output / "episodes").glob("*.json"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(list((output / "incomplete").glob("*.json"))), 1)
        episode = read_json(paths[0])
        self.assertEqual(episode["source_export"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        source_id = "src_" + hashlib.sha256(paths[0].read_bytes()).hexdigest()
        self.assertEqual(read_json(output / "split_manifest.json")["assignments"][source_id]["split"], "validation")
        with self.assertRaises(FileExistsError):
            export_histories([source], output)

    def test_export_rejects_invalid_shape_duplicate_runs_and_invalid_split(self):
        source = self.root / "histories.json"
        write_json(source, {"messages": []})
        with self.assertRaisesRegex(ValueError, "Not an ACP"):
            export_histories([source], self.root / "invalid")
        write_json(source, {**EXPORT, "runs": [history([]), history([])]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            export_histories([source], self.root / "duplicate")
        with self.assertRaisesRegex(ValueError, "split"):
            export_histories([source], self.root / "bad-split", split="unknown")

    def test_reconstruction_ingest_extract_and_raw_trace_retrieval_use_payload(self):
        source = self.root / "histories.json"
        write_json(source, {**EXPORT, "runs": [history([text("user_message_chunk", "Check repository."),
                                                       call(), result(rawOutput="working tree clean\n")])]})
        paths, manifest, _ = export_histories([source], self.root / "export")
        llm = ScriptedLLM({})
        recon = TraceReconstructor(llm, ReconstructionConfig(environment_id="acp.test", name="ACP"), self.root / "build")
        episodes, transitions = recon.ingest(paths, manifest)
        self.assertEqual(llm.calls, [])  # deterministic conversion prevents model annotation
        self.assertEqual(len(transitions), 1)
        payload = recon.extraction_payload(episodes[0], transitions[0])
        action = NormalizedAction.model_validate(payload["events"]["actions"][0]["content"])
        self.assertEqual(action.arguments["payload"]["title"], "git status")
        llm.responses["extract_transition"] = [LocalTransitionEvidence(
            id="placeholder", episode_id="placeholder", transition_id="placeholder", action=action,
            observation_text="must be replaced", confidence=Confidence(value=0.8, rationale="Observed"))]
        evidence = recon.extract_evidence(episodes, transitions)[0]
        self.assertEqual(evidence.action.arguments, action.arguments)
        self.assertEqual(evidence.observation_text, "working tree clean\n")
        corpus = TraceCorpus.build(paths, self.root / "corpus.sqlite")
        self.assertEqual(action_query(action), "execute git status")
        self.assertEqual(corpus.search("git")[0]["action"]["arguments"], action.arguments)

    def test_simulation_can_parameterize_state_changes_from_generic_payload(self):
        # Exercise the real compiler/harness against two converted writes, with no live LLM.
        artifacts = ReconstructionArtifacts(
            episodes=[], transitions=[], evidence=[],
            action_schema=ActionSchema(actions=[ActionSpec(name="execute", arguments={"payload": ArgumentSpec(type="object")})]),
            state_schema=StateSchema(fields=[StateField(path="world.files", type="object", default={})]),
            rules=[TransitionRule(id="write", action_type="execute", description="Recorded write",
                                  conditions=[Condition(left=Operand(action_arg="payload.kind"), op="eq", right=Operand(literal="edit"))],
                                  effects=[StateMutation(op="merge", path="world.files", value={
                                      "{action.arguments.payload.locations.0.path}": {"$action_arg": "payload.content.0.content.text"}})],
                                  renderer="write", observation_template="Wrote {action.arguments.payload.locations.0.path}")],
            renderers=[RenderContract(id="write", action_types=["execute"])])
        package = EnvironmentCompiler(ReconstructionConfig(
            environment_id="acp.write", name="Write", construction_kind="authored")).compile(artifacts, self.root / "package")
        harness = RuntimeHarness(package, self.root / "session")
        for written in ("first", "second"):
            initial = update("tool_call", toolCallId="write", title="Write x", kind="edit", status="in_progress",
                             locations=[{"path": "/x"}], content=[{"type": "content", "content": {"type": "text", "text": written}}])
            action = self.convert([initial, result("write", rawOutput="Wrote /x")])["events"][0]["content"]
            response = harness.step(StepRequest(action=NormalizedAction.model_validate(action)))
            self.assertEqual(response.observation, "Wrote /x")
            self.assertEqual(response.route, "rule")
            self.assertEqual(harness.session.load().world["files"]["/x"], written)


if __name__ == "__main__":
    unittest.main()
