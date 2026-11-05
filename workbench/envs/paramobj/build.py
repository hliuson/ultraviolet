"""Build param-object (wb_po_*) pools (docs/wb_paramobj.md).

  python -m workbench.envs.paramobj.build po --out DIR [--families a,b] [--tiers 0,1,2,3] [--per-cell 4000]

Candidate j of (family, tier) uses seed (crc32(family) + 1) * 10^8 + tier * 10^6 + j. Items are kept if
NV_MIN <= n_valid <= NV_MAX and guess_rate <= GUESS_MAX, and routed to eval / rl by paramobj.split_of (crc32 of the
canonical params, 20% eval). A cell stops at --per-cell rl rows or --max-draws candidates. Writes rl_train.jsonl,
eval_witness.jsonl and stats.json.
"""
import argparse
import json
import os
import zlib

from workbench.envs.paramobj import paramobj as po


def seed_of(family, tier, j):
    return (zlib.crc32(family.encode()) + 1) * 10 ** 8 + tier * 10 ** 6 + j


def build_cell(family, tier, per_cell, max_draws, eval_pct):
    out, n = {"rl": [], "eval": []}, {"rl": 0, "eval": 0}
    for j in range(max_draws):
        if len(out["rl"]) >= per_cell:
            break
        seed = seed_of(family, tier, j)
        p = po.sample_params(family, seed, tier)
        if p is None:
            continue
        keys, n_raw, gr, small = po.item_stats(family, p)
        if not (po.NV_MIN <= len(keys) <= po.NV_MAX) or gr > po.GUESS_MAX:
            continue
        split = po.split_of(family, p, eval_pct)
        out[split].append(po.make_row(family, p, tier, split, n[split], seed, keys, (n_raw, gr, small)))
        n[split] += 1
    return out


def cmd_po(a):
    fams = a.families.split(",") if a.families else sorted(po.FAMILIES)
    tiers = [int(t) for t in a.tiers.split(",")]
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    with open(os.path.join(a.out, "rl_train.jsonl"), "w") as frl, open(os.path.join(a.out, "eval_witness.jsonl"), "w") as fev:
        for fam in fams:
            for t in tiers:
                c = build_cell(fam, t, a.per_cell, a.max_draws, a.eval_pct)
                for r in c["rl"]:
                    frl.write(json.dumps(r) + "\n")
                for r in c["eval"][:a.eval_per_cell]:
                    fev.write(json.dumps(r) + "\n")
                stats[f"{fam[3:]}/{t}"] = {"rl": len(c["rl"]), "eval": min(len(c["eval"]), a.eval_per_cell)}
                print(fam, t, stats[f"{fam[3:]}/{t}"], flush=True)
    json.dump({"args": vars(a), "cells": stats}, open(os.path.join(a.out, "stats.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("po")
    b.add_argument("--out", required=True)
    b.add_argument("--families", default="", help="comma list of po_* names (default: all)")
    b.add_argument("--tiers", default="0,1,2,3")
    b.add_argument("--per-cell", type=int, default=4000, help="rl rows per (family, tier)")
    b.add_argument("--eval-per-cell", type=int, default=16)
    b.add_argument("--max-draws", type=int, default=100000)
    b.add_argument("--eval-pct", type=int, default=20)
    a = ap.parse_args(argv)
    cmd_po(a)


if __name__ == "__main__":
    main()
