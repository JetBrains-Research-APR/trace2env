#!/usr/bin/env python3
"""Search cross-fit report for one backbone directory (`work/exp-search-cv/backbones/<slug>/full458/`): the judged score of each
system present (by default only trace2env_v531), its five judge dimensions, a trajectory-clustered bootstrap interval of
the mean, the score per fold and by whether the row's application appears out of fold, resources, and the paired
difference against prompting when that baseline exists. Every row was predicted with the package built from the
trajectories outside its fold (k=5, `package_map.json`).

    python work/exp-search-cv/full458_report.py --backbone gpt-5.6-sol
"""
import argparse, glob, json, random, sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "work/exp-search-cv"
TERM = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(TERM / "awb"))
sys.path.insert(0, str(TERM / "backbones"))
from cluster_bootstrap import analyse  # noqa: E402
from compare_backbones import PRICE  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
SYSTEMS = ["prompting", "prompting_rag", "envpack_prompting", "harness_only_v51", "trace2env_v531"]


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(str(r["id"]), int(r["turn_idx"])): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())} if Path(path).exists() else {}


def cluster_ci(scores, rng, resamples=10000):
    """95% interval of the row mean under a bootstrap over trajectories (rows of one trajectory move together)."""
    by = {}
    for (tid, _), v in scores.items():
        by.setdefault(tid, []).append(v)
    tids = sorted(by)
    means = []
    for _ in range(resamples):
        picked = [by[tids[rng.randrange(len(tids))]] for _ in tids]
        rows = [v for group in picked for v in group]
        means.append(mean(rows))
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 2), "se": round(pstdev(d) / len(d) ** 0.5, 2) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|search|{name}"), 10000, 20000)
        out["ci"] = f"[{boot['ci95_rows'][0]:+.2f}, {boot['ci95_rows'][1]:+.2f}]"
        out["traj_wtl"] = f"{boot['trajectory_wins']}/{boot['trajectory_ties']}/{boot['trajectory_losses']}"
    return out


def fmt(p):
    if not p:
        return "—"
    return f"{p['mean']:+.2f} ± {p['se']:.2f} (n={p['rows']}; {p['wtl']}" + (f"; traj {p['ci']} {p['traj_wtl']}" if "ci" in p else "") + ")"


def usage_cost(u, price):
    return (u.get("prompt_tokens", 0) - u.get("cached_tokens", 0)) * price["prompt"] + u.get("cached_tokens", 0) * price["cached"] + u.get("completion_tokens", 0) * price["completion"]


def resources(directory, label, rows, price):
    infos = [(r.get("trace2env") or {}) for r in rows.values()]
    tool_calls = [len([t for t in (i.get("tool_calls") or []) if t != "_usage"]) for i in infos]
    latency = [i.get("latency_seconds") or 0 for i in infos]
    routes = {}
    for i in infos:
        routes[i.get("route")] = routes.get(i.get("route"), 0) + 1
    roles = {}
    for path in glob.glob(str(directory / f"calls-{label}-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = roles.setdefault(c.get("role"), {"calls": 0, "cache_hits": 0, "fresh_cost": [], "fresh_elapsed": [], "prompt": [], "errors": 0})
            r["calls"] += 1
            if c.get("error"):
                r["errors"] += 1
                continue
            if c.get("cache_hit"):
                r["cache_hits"] += 1
                continue
            u = c.get("usage") or {}
            r["fresh_elapsed"].append(float(c.get("elapsed_s") or 0))
            if u:
                r["fresh_cost"].append(usage_cost(u, price))
                r["prompt"].append(u.get("prompt_tokens") or 0)
    n = max(1, len(rows))
    per_role = {role: {"calls": r["calls"], "cache_hits": r["cache_hits"], "errors": r["errors"],
                       "cost_per_row": round((mean(r["fresh_cost"]) * (r["calls"] - r["errors"]) if r["fresh_cost"] else 0.0) / n, 3),
                       "time_per_row": round((mean(r["fresh_elapsed"]) * (r["calls"] - r["errors"]) if r["fresh_elapsed"] else 0.0) / n),
                       "prompt_mean": round(mean(r["prompt"])) if r["prompt"] else None, "prompt_max": max(r["prompt"]) if r["prompt"] else None}
                for role, r in roles.items()}
    return {"routes": routes, "tool_calls_per_row": round(mean(tool_calls), 2) if tool_calls else 0, "budget_rows": sum(c >= 8 for c in tool_calls),
            "latency_s_per_row": round(mean(latency)) if latency else None, "cost_per_row_total": round(sum(v["cost_per_row"] for v in per_role.values()), 3), "roles": per_role}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--rows", default=str(EXP / "full458_rows.jsonl"))
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    rows_in = keyed(args.rows)
    keys = sorted(rows_in)
    strata_in = json.load(open(EXP / "row_strata.json"))
    info = {k: strata_in[f"{k[0]}|{k[1]}"] for k in keys}
    groups = {"web_search rows": [k for k in keys if info[k]["tool"] == "web_search"], "web_extractor rows": [k for k in keys if info[k]["tool"] == "web_extractor"]}
    folds = {f"fold {f}": [k for k in keys if info[k]["fold"] == f] for f in range(5)}
    H = EXP / "backbones" / args.backbone / "full458"
    price = PRICE.get(args.backbone) or PRICE["gpt-5.6-sol"]
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; rows {len(keys)} in {len({k[0] for k in keys})} trajectories; strata (rows): "
          + ", ".join(f"{n} {len(v)}" for n, v in groups.items()) + "; folds (rows): " + ", ".join(f"{n} {len(v)}" for n, v in folds.items()) + "\n")
    names = list(groups)
    print("| system | valid | " + " | ".join(DIMS) + " | **total (458)** | clustered 95% CI | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(DIMS) + "---|---|" + "---|" * len(names))
    for name in SYSTEMS:
        path = H / f"judged-{name}.jsonl"
        if not path.exists():
            continue
        rows = keyed(path)
        rows_by[name] = {k: rows[k] for k in keys if k in rows}
        valid = {k: r for k, r in rows_by[name].items() if not r.get("failed")}
        scores[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
        dims = {d: mean(norm(r[d]) for r in valid.values()) for d in DIMS}
        lo, hi = cluster_ci(scores[name], random.Random(f"{args.seed}|{name}"))

        def m(ks):
            v = [scores[name][k] for k in ks if k in scores[name]]
            return f"{mean(v):.2f}" if v else "—"
        print(f"| {name} | {len(valid)}/{len(keys)} | " + " | ".join(f"{dims[d]:.2f}" for d in DIMS) + f" | **{m(keys)}** | [{lo:.2f}, {hi:.2f}] | " + " | ".join(m(groups[n]) for n in names) + " |")
    print("\nPer fold (total score)")
    print("| system | " + " | ".join(folds) + " |"); print("|---|" + "---|" * len(folds))
    for name in scores:
        print(f"| {name} | " + " | ".join(f"{mean([scores[name][k] for k in ks if k in scores[name]]):.2f}" if any(k in scores[name] for k in ks) else "—" for ks in folds.values()) + " |")
    pairs = [(a, b) for a, b in (("trace2env_v531", "prompting"), ("trace2env_v531", "harness_only_v51")) if a in scores and b in scores]
    if pairs:
        print("\nPaired differences (row mean ± SE; trajectory-clustered 95% CI; W/T/L)")
        for a, b in pairs:
            print(f"- {a} − {b}: {fmt(paired(scores[a], scores[b], keys, args.seed, f'{a}-{b}'))}")
    print("\nResources (fresh-equivalent cost at list price, per row)")
    for name in scores:
        print(f"- {name}: {json.dumps(resources(H, name, rows_by[name], price))}")


if __name__ == "__main__":
    main()
