"""JSONL storage: one JSON document per "\\n"-terminated line, whatever Unicode line separators the values contain."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces
from trace2env.storage import read_jsonl, write_jsonl


class JsonlTests(unittest.TestCase):
    def test_values_with_unicode_line_separators_round_trip(self):
        """str.splitlines() breaks at U+2028, U+2029 and U+0085; a search observation carried U+2028 live and the
        extraction stage could not read the ingested episodes back."""
        rows = [{"text": "line separator", "n": 1}, {"text": "nel\x85 and paragraph", "n": 2}, {"text": "\x1c\x1d\x1e", "n": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), len(rows))

    def test_jsonl_traces_with_unicode_line_separators_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            lines = ['{"episode_id": "a", "actor": "agent", "kind": "action", "content": {"type": "ls"}}',
                     '{"episode_id": "a", "actor": "tool", "content": "a b"}']
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            episodes = load_raw_traces(path)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(episodes[0].events), 2)
            self.assertIn("a b", [str(e.content) for e in episodes[0].events])


if __name__ == "__main__":
    unittest.main()
