"""Search cross-fit (k=5): trajectories, stratified folds, per-fold construction corpora, the trajectory -> package /
corpus maps and the shards. Rows are the benchmark's own search_test.jsonl (never modified); ids are unique in this
split, so the derived full458_rows.jsonl carries every field unchanged plus `benchmark_id`. Every trajectory is one
research question answered through `web_search` and `web_extractor` tool calls (no task instruction is shown: the
question is only implicit in the queries), so the trajectory is the task group; a row is predicted with the package
built from the trajectories outside its fold. Rows are stratified by the tool of the evaluated turn.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "work/exp-search-cv")
from trace2env.agentworld import load_rows, normalize_action, trajectory_episode  # noqa: E402
from trace2env.models import SplitAssignment, SplitManifest  # noqa: E402
from trace2env.storage import write_json  # noqa: E402
from trajectory_identity import cluster_trajectories, unique_id  # noqa: E402

ROOT = Path("work/exp-search-cv")
K = 5
SEED = 20261015
STYLE = "search"


def main() -> None:
    rows = load_rows(["work/agentworldbench/search_test.jsonl"])
    clusters = cluster_trajectories(rows)
    uid_of_row: dict[int, str] = {}
    trajectories: dict[str, dict] = {}
    for (task, tid, ordinal), members in clusters.items():
        uid = unique_id(task, tid, ordinal)
        longest = max(members, key=lambda r: len(r["prompt"]))
        first = normalize_action("search", longest["prompt"][0])
        trajectories[uid] = {
            "benchmark_id": tid, "style": STYLE, "rows": len(members), "turns": [int(r["turn_idx"]) for r in members],
            "visible_turns": len(longest["prompt"]), "total_turns": longest.get("total_turns"),
            "first_queries": [str(q)[:80] for q in (first.arguments.get("queries") or [])][:3],
            "tools": sorted({normalize_action("search", p).type for p in longest["prompt"]}),
        }
        for row in members:
            uid_of_row[id(row)] = uid
    derived = [{**row, "id": uid_of_row[id(row)], "benchmark_id": str(row["id"])} for row in rows]
    assert len(derived) == len(rows) and len(set(r["id"] for r in derived)) == len(trajectories)
    with open(ROOT / "full458_rows.jsonl", "w", encoding="utf-8") as f:
        for row in derived:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # groups: one trajectory = one question; folds balanced on visible turns
    rng = random.Random(SEED)
    fold_count: dict[int, int] = collections.Counter()
    fold_turns: dict[int, int] = collections.Counter()
    fold_of: dict[str, int] = {}
    for uid in sorted(trajectories, key=lambda u: (-trajectories[u]["visible_turns"], u)):
        keys = [(fold_turns[f], fold_count[f], rng.random()) for f in range(K)]
        fold = min(range(K), key=lambda f: keys[f])
        fold_of[uid] = fold
        fold_count[fold] += 1
        fold_turns[fold] += trajectories[uid]["visible_turns"]
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
            path = out / f"search_{uid}.json"
            write_json(path, episode)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assignments[f"src_{digest}"] = SplitAssignment(split="train", trajectory_group=episode["trajectory_group"])
        write_json(out.parent / "split_manifest.json", SplitManifest(assignments=assignments).model_dump(mode="json"))
        corpora[f"{STYLE}-f{fold}"] = {"style": STYLE, "fold": fold, "trajectories": members, "count": len(members),
                                       "transitions": sum(trajectories[u]["visible_turns"] for u in members)}
    shards = ROOT / "full458_shards"
    shards.mkdir(exist_ok=True)
    for old in shards.glob("traj_*.jsonl"):
        old.unlink()
    for uid in trajectories:
        with open(shards / f"traj_{uid}.jsonl", "w", encoding="utf-8") as f:
            for row in sorted((r for r in derived if r["id"] == uid), key=lambda r: int(r["turn_idx"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(ROOT / "full458_rows_ids.json", [{"id": r["id"], "turn_idx": r["turn_idx"]} for r in derived])
    write_json(ROOT / "trajectories.json", trajectories)
    write_json(ROOT / "corpora.json", corpora)
    write_json(ROOT / "package_map.json", {uid: t["package"] for uid, t in trajectories.items()})
    write_json(ROOT / "package_map_schema.json", {uid: t["package"] + "-schema_only" for uid, t in trajectories.items()})
    write_json(ROOT / "corpus_map.json", {uid: t["corpus"] for uid, t in trajectories.items()})

    strata = {}
    for row in derived:
        t = trajectories[row["id"]]
        strata[f"{row['id']}|{row['turn_idx']}"] = {"style": STYLE, "fold": t["fold"], "tool": normalize_action("search", row["current_prompt"]).type}
    write_json(ROOT / "row_strata.json", strata)
    print(f"rows {len(derived)} | trajectories {len(trajectories)}")
    print("  per fold: " + ", ".join(f"f{f}={sum(1 for t in trajectories.values() if t['fold'] == f)} traj/"
                                      f"{sum(t['rows'] for t in trajectories.values() if t['fold'] == f)} rows/"
                                      f"{sum(t['visible_turns'] for t in trajectories.values() if t['fold'] == f)} turns" for f in range(K)))
    print("  corpora: " + ", ".join(f"{k}={v['count']} traj/{v['transitions']} transitions" for k, v in corpora.items()))
    print("  rows by tool:", dict(collections.Counter(s["tool"] for s in strata.values())))
    for fold in range(K):
        for u, t in trajectories.items():
            assert u not in corpora[f"{STYLE}-f{fold}"]["trajectories"] or t["fold"] != fold, "leak: a trajectory in its own fold's corpus"
    print("  leak check: no trajectory is in the corpus of its own fold")


if __name__ == "__main__":
    main()
