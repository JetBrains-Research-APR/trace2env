"""OS cross-fit (k=5): trajectories, task-grouped stratified folds, per-fold construction corpora, the trajectory ->
package / corpus maps and the shards. Rows are the benchmark's own os_test.jsonl (never modified); ids are unique in this
split, so the derived full200_rows.jsonl carries every field unchanged plus `benchmark_id`. Every trajectory is a desktop
computer-use session (OSWorld-style tasks on Thunderbird, LibreOffice, VLC, Chrome, GIMP, VS Code, the file manager and the
terminal, observed as accessibility trees); a row is predicted with the package built from the trajectories outside its
fold. Trajectories with the same task instruction share a fold, so a task is never both constructed from and evaluated.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "work/exp-os-cv")
from trace2env.agentworld import load_rows, normalize_action, trajectory_episode  # noqa: E402
from trace2env.models import SplitAssignment, SplitManifest  # noqa: E402
from trace2env.storage import write_json  # noqa: E402
from trajectory_identity import cluster_trajectories, unique_id  # noqa: E402

ROOT = Path("work/exp-os-cv")
K = 5
SEED = 20261015
STYLE = "os"
APPS = ("Thunderbird", "LibreOffice Calc", "LibreOffice Writer", "LibreOffice Impress", "LibreOffice", "VLC", "Chrome", "Chromium",
        "GIMP", "VS Code", "Visual Studio Code", "terminal", "file manager", "Files")


def instruction_of(prompt: str) -> str:
    m = re.search(r"\*\*Task Instruction:\*\*\s*(.+?)(?:\n\n\*\*|\Z)", prompt, re.S)
    return (m.group(1) if m else prompt[:400]).strip()


def apps_of(text: str) -> list[str]:
    low = text.lower()
    return sorted({app for app in APPS if app.lower() in low})


def main() -> None:
    rows = load_rows(["work/agentworldbench/os_test.jsonl"])
    clusters = cluster_trajectories(rows)
    uid_of_row: dict[int, str] = {}
    trajectories: dict[str, dict] = {}
    for (task, tid, ordinal), members in clusters.items():
        uid = unique_id(task, tid, ordinal)
        longest = max(members, key=lambda r: len(r["prompt"]))
        instruction = instruction_of(longest["prompt"][0])
        trajectories[uid] = {
            "benchmark_id": tid, "style": STYLE, "rows": len(members), "turns": [int(r["turn_idx"]) for r in members],
            "visible_turns": len(longest["prompt"]), "total_turns": longest.get("total_turns"),
            "instruction": instruction[:200], "instruction_sha": hashlib.sha256(instruction.encode()).hexdigest()[:12],
            "apps": apps_of(instruction), "tools": sorted({normalize_action("os", p).type for p in longest["prompt"]}),
        }
        for row in members:
            uid_of_row[id(row)] = uid
    derived = [{**row, "id": uid_of_row[id(row)], "benchmark_id": str(row["id"])} for row in rows]
    assert len(derived) == len(rows) and len(set(r["id"] for r in derived)) == len(trajectories)
    with open(ROOT / "full200_rows.jsonl", "w", encoding="utf-8") as f:
        for row in derived:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # groups: identical task instruction (the same OSWorld task solved more than once) never straddles a fold
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for uid, t in trajectories.items():
        groups[f"task:{t['instruction_sha']}"].append(uid)
    rng = random.Random(SEED)
    fold_count: dict[int, int] = collections.Counter()
    fold_turns: dict[int, int] = collections.Counter()
    fold_of: dict[str, int] = {}
    ordered = sorted(groups.items(), key=lambda kv: (-sum(trajectories[u]["visible_turns"] for u in kv[1]), kv[0]))
    for g, uids in ordered:
        keys = [(fold_turns[f], fold_count[f], rng.random()) for f in range(K)]
        fold = min(range(K), key=lambda f: keys[f])
        for u in uids:
            fold_of[u] = fold
            fold_count[fold] += 1
            fold_turns[fold] += trajectories[u]["visible_turns"]
    for uid, t in trajectories.items():
        t["fold"] = fold_of[uid]
        t["package"] = f"ws/{STYLE}-f{t['fold']}/packages/{STYLE}-f{t['fold']}"
        t["corpus"] = f"corpora/{STYLE}-f{t['fold']}/episodes"

    longest_rows: dict[str, dict] = {}
    for row in derived:
        if row["id"] not in longest_rows or len(row["prompt"]) > len(longest_rows[row["id"]]["prompt"]):
            longest_rows[row["id"]] = row
    corpora = {}
    for fold in range(K):
        members = sorted(u for u, t in trajectories.items() if t["fold"] != fold)
        out = ROOT / "corpora" / f"{STYLE}-f{fold}" / "episodes"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.json"):
            old.unlink()
        assignments = {}
        for uid in members:
            episode = trajectory_episode(longest_rows[uid])
            episode["split"] = "train"
            path = out / f"os_{uid}.json"
            write_json(path, episode)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assignments[f"src_{digest}"] = SplitAssignment(split="train", trajectory_group=episode["trajectory_group"])
        write_json(out.parent / "split_manifest.json", SplitManifest(assignments=assignments).model_dump(mode="json"))
        corpora[f"{STYLE}-f{fold}"] = {"style": STYLE, "fold": fold, "trajectories": members, "count": len(members),
                                       "transitions": sum(trajectories[u]["visible_turns"] for u in members)}
    shards = ROOT / "full200_shards"
    shards.mkdir(exist_ok=True)
    for old in shards.glob("traj_*.jsonl"):
        old.unlink()
    for uid in trajectories:
        with open(shards / f"traj_{uid}.jsonl", "w", encoding="utf-8") as f:
            for row in sorted((r for r in derived if r["id"] == uid), key=lambda r: int(r["turn_idx"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(ROOT / "full200_rows_ids.json", [{"id": r["id"], "turn_idx": r["turn_idx"]} for r in derived])
    write_json(ROOT / "trajectories.json", trajectories)
    write_json(ROOT / "corpora.json", corpora)
    write_json(ROOT / "package_map.json", {uid: t["package"] for uid, t in trajectories.items()})
    write_json(ROOT / "package_map_schema.json", {uid: t["package"] + "-schema_only" for uid, t in trajectories.items()})
    write_json(ROOT / "corpus_map.json", {uid: t["corpus"] for uid, t in trajectories.items()})

    # strata: the row's application also appears in an out-of-fold trajectory (the knowledge the package can carry over)
    strata = {}
    for row in derived:
        t = trajectories[row["id"]]
        pool = [x for u, x in trajectories.items() if x["fold"] != t["fold"]]
        shared = bool(set(t["apps"]) & {a for x in pool for a in x["apps"]})
        strata[f"{row['id']}|{row['turn_idx']}"] = {"style": STYLE, "fold": t["fold"], "apps": t["apps"], "app_shared": shared}
    write_json(ROOT / "row_strata.json", strata)
    print(f"rows {len(derived)} | trajectories {len(trajectories)} | task groups {len(groups)} "
          f"(groups with >1 trajectory: {sum(1 for g in groups.values() if len(g) > 1)})")
    print("  per fold: " + ", ".join(f"f{f}={sum(1 for t in trajectories.values() if t['fold'] == f)} traj/"
                                      f"{sum(t['rows'] for t in trajectories.values() if t['fold'] == f)} rows/"
                                      f"{sum(t['visible_turns'] for t in trajectories.values() if t['fold'] == f)} turns" for f in range(K)))
    print("  corpora: " + ", ".join(f"{k}={v['count']} traj/{v['transitions']} transitions" for k, v in corpora.items()))
    print("  rows whose app appears out of fold:", sum(1 for s in strata.values() if s["app_shared"]), "of", len(strata))
    for fold in range(K):
        for u, t in trajectories.items():
            assert u not in corpora[f"{STYLE}-f{fold}"]["trajectories"] or t["fold"] != fold, "leak: a trajectory in its own fold's corpus"
    print("  leak check: no trajectory is in the corpus of its own fold")


if __name__ == "__main__":
    main()
