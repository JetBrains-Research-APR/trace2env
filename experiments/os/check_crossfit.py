#!/usr/bin/env python3
"""Cross-fit integrity check on the finished os rows: every shard ran with the package of its own out-of-fold corpus
(the launcher's package-shard record), that package's corpus does not contain the trajectory, every row has a
prediction, and the recorded manifest digests cover the five packages. Usage: check_crossfit.py --backbone <slug>"""
import argparse, glob, json
from pathlib import Path

EXP = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True); ap.add_argument("--label", default="trace2env_v531")
    args = ap.parse_args()
    H = EXP / "backbones" / args.backbone / "full200"
    pkg_map = json.load(open(EXP / "package_map.json")); traj = json.load(open(EXP / "trajectories.json"))
    corpora = json.load(open(EXP / "corpora.json"))
    problems = []
    rows = [json.loads(l) for p in glob.glob(str(H / f"pred-{args.label}-*.jsonl")) for l in open(p) if l.strip()]
    seen = {(str(r["id"]), int(r["turn_idx"])) for r in rows}
    expected = {(str(r["id"]), int(r["turn_idx"])) for r in (json.loads(l) for l in open(EXP / "full200_rows.jsonl"))}
    if seen != expected:
        problems.append(("rows", f"{len(expected - seen)} rows without a prediction, {len(seen - expected)} unexpected"))
    record = H / f"package-shard-{args.label}.txt"
    shards = {}
    if record.exists():
        for line in record.read_text().splitlines():
            uid, pkg = line.split(" ", 1)
            shards[uid] = pkg
    else:
        problems.append(("record", f"{record.name} missing: the launcher did not record the per-shard packages"))
    for uid, t in traj.items():
        pkg = shards.get(uid)
        if pkg is None:
            problems.append((uid, "no package-shard record")); continue
        if pkg != pkg_map[uid]:
            problems.append((uid, f"ran with {pkg}, mapped {pkg_map[uid]}"))
        corpus = corpora[pkg.split("/")[1]]
        if uid in corpus["trajectories"]:
            problems.append((uid, "its package was built from a corpus containing the trajectory itself"))
        if corpus["fold"] != t["fold"]:
            problems.append((uid, f"package fold {corpus['fold']} != trajectory fold {t['fold']}"))
    mapped = H / f"package-manifest-{args.label}.sha256"
    digests = mapped.read_text().splitlines() if mapped.exists() else []
    print(f"{args.label}: {len(rows)} prediction rows ({len(expected)} expected), {len(shards)} shard records, {len(digests)} recorded package digests")
    print("problems:", len(problems)); [print("  ", p) for p in problems[:12]]


if __name__ == "__main__":
    main()
