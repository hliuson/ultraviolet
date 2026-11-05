"""sft.py with block-structured epochs (as used for ultraviolet-50m-mid-alpha). sft.py itself is not modified; this imports its loader / batching /
logging helpers (load_examples, heldout_examples, make_xy, micro_batches, epoch_plan, TRSft) and re-implements only
main() with these differences:

  * --traces is ONE file whose rows carry `block` (1..B), built by the data-prep step: block b = a shuffled copy of the
    Violetto slice + 1/B of the synthetic rows. sft.load_examples runs per block (its exact-duplicate filter would
    otherwise drop copies 2..B of the repeated slice). Each block is one "epoch" of sft.py: sft.epoch_plan(n_b, gb,
    seed, b-1, lens_b) (shuffle, length-sort in mega-chunks, shuffle batch order) over that block's rows only.
  * The LR schedule (warmup --warmup-frac, then linear decay to 0) runs over the whole run: total = sum of the blocks'
    steps. epoch{b}.pt is saved at the end of block b (same contents as sft.py's per-epoch checkpoints).
  * Held-out NLL (response tokens, same as sft.py) is computed per group of the held-out file (task numina_math ->
    m1_math, other tasks -> m1_tasks, VS / MP / AR1 as is), plus m1_all (the m1_ groups together) and all (every held-out row; logged as sum/heldout_nll_mix like sft.py --tr-stage mid).
  * --nll-only CKPT...: no training; write the per-group held-out NLL of each checkpoint to --nll-out.
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import sft  # noqa: E402
from ultraviolet.model import MathConfig, MathLM  # noqa: E402
from ultraviolet.optim import apply_schedule, build_optimizers  # noqa: E402

M1_TASKS_MATH = ("numina_math",)


def group_of(r):
    t = r.get("task")
    if t in ("VS", "MP", "AR1", "VC"):
        return t
    return "m1_math" if t in M1_TASKS_MATH else "m1_tasks"


def heldout_groups(path, tk, max_len):
    rows = [json.loads(l) for l in open(path)]
    vex = sft.heldout_examples(path, tk, max_len)
    g = collections.defaultdict(list)
    for r, e in zip(rows, vex):
        g[group_of(r)].append(e)
    return rows, dict(g)


def make_evaluator(raw, groups, args, dev, amp, rank, world, ddp):
    vmb = {k: sft.micro_batches(v, args.tok_budget, args.pad_to)[rank::world] for k, v in sorted(groups.items())}

    def evaluate():
        raw.eval()
        tot = torch.zeros(len(vmb), device=dev, dtype=torch.float64)
        n = torch.zeros(len(vmb), device=dev, dtype=torch.float64)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            for gi, mbs in enumerate(vmb.values()):
                for mb in mbs:
                    x, y = sft.make_xy(mb, args.pad_to)
                    x, y = x.to(dev), y.to(dev)
                    k = (y != -1).sum()
                    tot[gi] += raw(x, y).double() * k
                    n[gi] += k
        if ddp:
            dist.all_reduce(tot)
            dist.all_reduce(n)
        raw.train()
        names = list(vmb)
        out = {g: (tot[i] / n[i]).item() for i, g in enumerate(names)}
        m1 = [i for i, g in enumerate(names) if g.startswith("m1_")]
        out["m1_all"] = (tot[m1].sum() / n[m1].sum()).item()
        out["all"] = (tot.sum() / n.sum()).item()
        out["_tokens"] = {g: int(n[i].item()) for i, g in enumerate(names)}
        return out
    return evaluate


def load_model(path, dev):
    s = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MathConfig(**s["cfg"])
    raw = MathLM(cfg)
    raw.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    return raw.to(dev), cfg, s


def main():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--init", default="")
    a("--name", default="")
    a("--out-root", default="")
    a("--traces", default="")
    a("--heldout", required=True)
    a("--tok", required=True)
    a("--global-batch", type=int, default=32)
    a("--tok-budget", type=int, default=16384)
    a("--max-len", type=int, default=4096)
    a("--pad-to", type=int, default=512)
    a("--lr-mult", type=float, default=0.15)
    a("--warmup-frac", type=float, default=0.05)
    a("--val-every", type=int, default=250)
    a("--seed", type=int, default=0)
    a("--compile", type=int, default=1)
    a("--wandb-mode", default="auto")
    a("--tr-name", default="")
    a("--tr-stage", default="mid", choices=["sft", "mid"])
    a("--tr-parent", default="")
    a("--tr-data", default="")
    a("--tr-mix", default="")
    a("--nll-only", nargs="*", default=None, help="checkpoints: per-group held-out NLL only, no training")
    a("--nll-out", default="")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, lrank = 0, 1, 0
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    master = rank == 0
    amp = torch.float16 if torch.cuda.get_device_capability(dev)[0] < 8 else torch.bfloat16
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tok)
    held_rows, groups = heldout_groups(args.heldout, tk, args.max_len)
    held_pids = {r["problem_id"] for r in held_rows}

    if args.nll_only is not None:
        res = {}
        for c in args.nll_only:
            raw, _, _ = load_model(c, dev)
            res[c] = make_evaluator(raw, groups, args, dev, amp, rank, world, ddp)()
            if master:
                print("NLL", c, json.dumps(res[c]), flush=True)
            del raw
        if master and args.nll_out:
            json.dump(res, open(args.nll_out, "w"), indent=1)
        return

    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    # ------------------------------------------------------------- per-block loading
    by_block = collections.defaultdict(list)
    for line in open(args.traces):
        by_block[json.loads(line)["block"]].append(line)
    tmpd = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"sftblk-{args.name}-{rank}")
    os.makedirs(tmpd, exist_ok=True)
    ex, blocks, drops = [], [], {}
    for b in sorted(by_block):
        p = os.path.join(tmpd, f"block{b}.jsonl")
        with open(p, "w") as f:
            f.writelines(by_block[b])
        e, d = sft.load_examples([p], tk, held_pids, args.max_len)
        drops[b] = dict(d, rows=len(by_block[b]), examples=len(e))
        blocks.append((len(ex), len(e)))
        ex += e
        os.remove(p)
    del by_block
    lens = np.array([len(t) for t, _ in ex])

    raw, cfg, s = load_model(args.init, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    model = raw
    if args.compile:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        model = torch.compile(raw, dynamic=False)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrank])
    mu, adam = build_optimizers(raw, lr_mult=args.lr_mult, distributed=ddp)

    gb = args.global_batch
    blk_steps = [math.ceil(n / gb) for _, n in blocks]
    total = sum(blk_steps)
    warm = max(1, int(args.warmup_frac * total))
    run_cfg = {"model": cfg.to_dict(), "train_examples": len(ex), "blocks": blocks, "block_steps": blk_steps,
               "loader": drops, "heldout_groups": {k: len(v) for k, v in groups.items()},
               "train_tokens": int(lens.sum()), "train_response_tokens": int(sum(len(t) - s_ for t, s_ in ex)),
               "total_steps": total, "warmup_steps": warm, "world": world, "amp": str(amp), "init_step": s.get("step")}
    del s
    wb = None
    if master:
        print(json.dumps(run_cfg, default=str), flush=True)
        json.dump({"args": vars(args), **run_cfg}, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
        if args.tr_name and args.wandb_mode != "disabled":
            wb = sft.TRSft(args, run_cfg, out)
    evaluate = make_evaluator(raw, groups, args, dev, amp, rank, world, ddp)
    best = float("inf")
    hist = []

    def log_val(step, epoch):
        nonlocal best
        v = evaluate()
        best = min(best, v["all"])
        if master:
            print(f"VAL step {step} epoch {epoch:.3f} heldout_resp_nll {v['all']:.4f} min {best:.4f} "
                  + " ".join(f"{g} {x:.4f}" for g, x in v.items() if g not in ("all", "_tokens")), flush=True)
            rec = {"step": step, "epoch": epoch, "val_nll": v["all"], "val_nll_min": best,
                   "groups": {g: x for g, x in v.items() if g not in ("all", "_tokens")}}
            hist.append(rec)
            with open(os.path.join(out, "val.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if wb:
                wb.log({"step": step, "epoch": epoch, "val/resp_nll": v["all"], "val/resp_nll_min": best})
                kv = {f"task/{g}/nll": x for g, x in rec["groups"].items() if g != "m1_all"}
                kv["diag/heldout_nll_m1_all"] = rec["groups"]["m1_all"]
                wb.tl.log(kv, step=step)

    log_val(0, 0.0)
    raw.train()
    step = 0
    t0 = time.time()
    for bi, (off, nb) in enumerate(blocks):
        plan = sft.epoch_plan(nb, gb, args.seed, bi, lens[off:off + nb])
        assert len(plan) == blk_steps[bi]
        for bidx in plan:
            bidx = bidx + off
            if len(bidx) < world:
                continue
            gbatch = [ex[i] for i in sorted(bidx, key=lambda i: lens[i])]
            n_resp = sum(len(t) - s_ for t, s_ in gbatch)
            mbs = sft.micro_batches(gbatch[rank::world], args.tok_budget, args.pad_to)
            apply_schedule(mu, adam, step, total, warm, decay_frac=1.0, mom_warmup=1, mom0=0.95, mom1=0.95)
            lsum = torch.zeros((), device=dev)
            for i, mb in enumerate(mbs):
                x, y = sft.make_xy(mb, args.pad_to, args.tok_budget)
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                k = (y != -1).sum()
                last = i == len(mbs) - 1
                ctx = model.no_sync() if (ddp and not last) else torch.enable_grad()
                with ctx, torch.autocast("cuda", dtype=amp):
                    loss = model(x, y) * k / n_resp * world
                    scaler.scale(loss).backward()
                lsum += loss.detach() / world
            scaler.unscale_(mu)
            scaler.unscale_(adam)
            scaler.step(mu)
            scaler.step(adam)
            scaler.update()
            mu.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
            step += 1
            if ddp:
                dist.all_reduce(lsum)
            ep_f = bi + (step - sum(blk_steps[:bi])) / blk_steps[bi]
            if master and (step % 10 == 0 or step == 1):
                el = time.time() - t0
                print(f"step {step}/{total} ep {bi} loss {lsum.item():.4f} lr_f "
                      f"{adam.param_groups[0]['lr'] / adam.param_groups[0]['base_lr']:.3f} "
                      f"{el / step:.2f}s/step scale {scaler.get_scale():.0f}", flush=True)
                if wb:
                    wb.log({"step": step, "epoch": ep_f, "train/loss": lsum.item()})
            if step % args.val_every == 0:
                log_val(step, ep_f)
        if step % args.val_every:
            log_val(step, float(bi + 1))
        if master:
            p = os.path.join(out, f"epoch{bi + 1}.pt")
            torch.save({"model": raw.state_dict(), "cfg": cfg.to_dict(), "step": step, "epoch": bi + 1,
                        "init": os.path.abspath(args.init), "val_nll": hist[-1]["val_nll"],
                        "val_groups": hist[-1]["groups"]}, p + ".tmp")
            os.replace(p + ".tmp", p)
            print(f"saved {p}", flush=True)
    if master:
        res = {"name": args.name, "final_val_nll": hist[-1]["val_nll"], "min_val_nll": best,
               "init_val_nll": hist[0]["val_nll"], "steps": total, "minutes": round((time.time() - t0) / 60, 1),
               "final_groups": hist[-1]["groups"], "init_groups": hist[0]["groups"]}
        json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
        print("RESULT", json.dumps(res), flush=True)
        if wb:
            wb.summary.update({k: v for k, v in res.items() if not isinstance(v, dict)})
            wb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
