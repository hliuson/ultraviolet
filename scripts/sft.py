"""SFT of a pretrained math LM on thinking traces (full fine-tune).

The format matches pretraining: `Problem:\\n{problem}\\n\\n<think>\\n` (prompt, no loss) + `{thinking}\\n</think>\\n\\n
{solution}<eos>` (response, loss). Prompt and response are tokenized separately and concatenated, which matches
the generation prompt used by eval_math.py. Each example gets its own row (no packing, so no cross-example
attention). Rows are right-padded and length-bucketed: each global batch takes similar-length examples from a
shuffled mega-chunk. Loss is the sum of response-token NLL / response tokens in the global batch.

The optimizer is NorMuon + AdamW groups (optim.py) at --lr-mult x the pretraining LRs, with linear warmup and then
linear decay to 0. One checkpoint per epoch (weights only). Held-out trace response NLL is logged every --val-every
steps, with its min-so-far.

Held-out split: every problem_id in --heldout (prep_traces.py's heldout_traces.jsonl; the same split as the
pretraining held-out traces) is excluded from training, and those traces form the val set. --traces takes several jsonl files, so new
trace batches can be appended.

  torchrun --standalone --nproc_per_node 2 scripts/sft.py --init RUN/final.pt --name NAME \\
      --traces a.jsonl b.jsonl --heldout vtr_v0/heldout_traces.jsonl --tok tokenizer.json --epochs 4
"""
import argparse
import json
import types
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from prep_traces import prompt_text, render  # noqa: E402
from ultraviolet.model import MathConfig, MathLM  # noqa: E402
from ultraviolet.optim import apply_schedule, build_optimizers  # noqa: E402

EOS = 0
PROJECT, GROUP = os.environ.get("WANDB_PROJECT", "ultraviolet"), "math-sft"


def load_examples(files, tk, heldout_pids, max_len):
    tr, n_long, n_bad, seen = [], 0, 0, set()
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("problem_id") in heldout_pids:
                continue
            if "correct" in r and str(r["correct"]).lower() not in ("true", "1"):
                continue
            if r.get("format") == "plain":  # plain-text row: [EOS] + text + [EOS], loss on every token
                key = (r.get("problem_id"), hash(r["text"]))
                if key not in seen:
                    seen.add(key)
                    tr.append(("", r["text"]))
                continue
            prob = r.get("problem") or r.get("prompt")
            out = render(prob, r["trace"])
            if out is None:
                n_bad += 1
                continue
            key = (r.get("problem_id"), hash(out[1]), hash(out[2]))  # exact duplicates only
            if key in seen:
                continue
            seen.add(key)
            tr.append((prompt_text(prob), f"{out[1]}\n</think>\n\n{out[2]}"))
    enc_p = tk.encode_batch([p for p, _ in tr])
    enc_r = tk.encode_batch([r for _, r in tr])
    ex = []
    for p, r in zip(enc_p, enc_r):
        ids = [EOS] + p.ids + r.ids + [EOS]
        if len(ids) > max_len + 1:
            n_long += 1
            continue
        ex.append((np.array(ids, dtype=np.int64), 1 + len(p.ids)))  # (tokens, index of first response token)
    return ex, {"unclosed_or_empty": n_bad, "too_long": n_long}


def heldout_examples(path, tk, max_len):
    ex = []
    for line in open(path):
        r = json.loads(line)
        if r.get("format") == "plain":  # plain-text row (some held-out rows also carry a text field)
            p, q = [], tk.encode(r["text"]).ids
        else:
            p = tk.encode(prompt_text(r["problem"])).ids
            q = tk.encode(f"{r['thinking']}\n</think>\n\n{r['solution']}").ids
        ids = [EOS] + p + q + [EOS]
        ex.append((np.array(ids[:max_len + 1], dtype=np.int64), 1 + len(p)))
    return ex


def make_xy(batch, pad_to, budget=None):
    """Right-padded (x, y). With `budget`, the row count is padded to budget // L (empty rows have no targets), so
    there is one static shape per length bucket (torch.compile does not recompile per batch)."""
    L = max(len(t) for t, _ in batch) - 1
    L = ((L + pad_to - 1) // pad_to) * pad_to
    R = max(len(batch), budget // L) if budget else len(batch)
    x = np.full((R, L), EOS, dtype=np.int64)
    y = np.full((R, L), -1, dtype=np.int64)
    for i, (t, s) in enumerate(batch):
        n = len(t) - 1
        x[i, :n] = t[:-1]
        y[i, s - 1:n] = t[s:]  # target at position j is token j+1; response tokens are t[s:]
    return torch.from_numpy(x), torch.from_numpy(y)


def micro_batches(batch, budget, pad_to):
    """Split a (length-sorted) list into micro-batches with B * padded_len <= budget."""
    out, cur = [], []
    for e in sorted(batch, key=lambda e: len(e[0])):
        L = ((len(e[0]) - 1 + pad_to - 1) // pad_to) * pad_to
        if cur and (len(cur) + 1) * L > budget:
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def epoch_plan(n, gb, seed, epoch, lens, chunk_mult=8):
    """Global batches (lists of example indices) for one epoch: shuffle, then sort by length within mega-chunks of
    gb*chunk_mult, cut into batches of gb, and shuffle the batch order."""
    rng = np.random.default_rng([seed, epoch])
    perm = rng.permutation(n)
    batches = []
    for i in range(0, n, gb * chunk_mult):
        ch = perm[i:i + gb * chunk_mult]
        ch = ch[np.argsort(lens[ch], kind="stable")]
        batches += [ch[j:j + gb] for j in range(0, len(ch), gb)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


class TRSft:
    """wb.log/summary/finish calls routed to tr_log."""
    def __init__(self, args, run_cfg, out):
        import tr_log
        self.tl, self.mix_key = tr_log, args.tr_stage == "mid"
        mix = None
        if args.tr_mix:
            mix = json.load(open(args.tr_mix)) if os.path.exists(args.tr_mix) else json.loads(args.tr_mix)
        tr_log.init(args.tr_name, args.tr_stage, parent=args.tr_parent or args.init, data=args.tr_data or None,
                    mix=mix, config={"args": vars(args), **run_cfg}, ckpt_path=out, dir=out)
        self.summary = types.SimpleNamespace(update=lambda d: tr_log.summary({f"diag/{k}": v for k, v in d.items()}))

    def log(self, d):
        step = d["step"]
        if "val/resp_nll" in d:
            kv = self.tl.sft_val_keys({"val_nll": d["val/resp_nll"], "val_nll_min": d["val/resp_nll_min"],
                                       "epoch": d["epoch"]})
            if self.mix_key:
                kv["sum/heldout_nll_mix"] = kv.pop("sum/heldout_nll")
        else:
            kv = {"train/loss": d["train/loss"], "train/epoch": d["epoch"]}
        self.tl.log(kv, step=step)

    def finish(self):
        self.tl.finish()


def main():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--init", required=True, help="pretrained checkpoint (ckpt.pt / final.pt)")
    a("--name", required=True)
    a("--out-root", default="runs/sft")
    a("--traces", nargs="+", required=True)
    a("--heldout", required=True, help="heldout_traces.jsonl (its problem_ids are excluded from training)")
    a("--tok", required=True)
    a("--epochs", type=int, default=4)
    a("--global-batch", type=int, default=32, help="examples per optimizer step")
    a("--tok-budget", type=int, default=16384, help="max B * padded_len per micro-batch")
    a("--max-len", type=int, default=4096)
    a("--pad-to", type=int, default=512)
    a("--lr-mult", type=float, default=0.15, help="x pretraining LRs (NorMuon 0.018, Adam groups Table 1)")
    a("--warmup-frac", type=float, default=0.05)
    a("--val-every", type=int, default=25)
    a("--seed", type=int, default=0)
    a("--compile", type=int, default=1)
    a("--wandb-mode", default="auto")
    a("--tr-name", default="", help="log through tr_log under this id instead of a plain wandb run")
    a("--tr-stage", default="sft", choices=["sft", "mid"], help="mid: held-out NLL goes to sum/heldout_nll_mix")
    a("--tr-parent", default="", help="tr_log parent run id, e.g. the pretrain run")
    a("--tr-data", default="", help="tr_log data id (pack)")
    a("--tr-mix", default="", help="JSON file or string: per-task token shares")
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
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tok)
    held = [json.loads(l) for l in open(args.heldout)]
    held_pids = {r["problem_id"] for r in held}
    ex, drop = load_examples(args.traces, tk, held_pids, args.max_len)
    vex = heldout_examples(args.heldout, tk, args.max_len)
    lens = np.array([len(t) for t, _ in ex])

    s = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = MathConfig(**s["cfg"])
    raw = MathLM(cfg)
    raw.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    raw = raw.to(dev)
    amp = torch.float16 if torch.cuda.get_device_capability(dev)[0] < 8 else torch.bfloat16
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
    steps_per_epoch = math.ceil(len(ex) / gb)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_frac * total))
    resp_tokens = int(sum(len(t) - s_ for t, s_ in ex))
    run_cfg = {"args": vars(args), "model": cfg.to_dict(), "train_examples": len(ex), "dropped": drop,
               "heldout_examples": len(vex), "heldout_problems": len(held_pids), "train_tokens": int(lens.sum()),
               "train_response_tokens": resp_tokens, "steps_per_epoch": steps_per_epoch, "total_steps": total,
               "world": world, "init_step": s.get("step")}
    wb = None
    if master:
        print(json.dumps(run_cfg, default=str), flush=True)
        json.dump(run_cfg, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
        if args.tr_name and args.wandb_mode != "disabled":
            wb = TRSft(args, run_cfg, out)
        elif args.wandb_mode != "disabled":
            import wandb
            wb = wandb.init(project=PROJECT, group=GROUP, name=args.name, id=f"msft-{args.name}", resume="allow",
                            config=run_cfg, dir=out, mode="online" if args.wandb_mode == "auto" else args.wandb_mode,
                            settings=wandb.Settings(silent=True, init_timeout=90))
            wb.define_metric("step")
            wb.define_metric("*", step_metric="step")

    vmb = micro_batches(vex, args.tok_budget, args.pad_to)[rank::world]

    def evaluate():
        raw.eval()
        tot = torch.zeros((), device=dev)
        n = torch.zeros((), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            for mb in vmb:
                x, y = make_xy(mb, args.pad_to)
                x, y = x.to(dev), y.to(dev)
                k = (y != -1).sum()
                tot += raw(x, y).float() * k
                n += k
        if ddp:
            dist.all_reduce(tot)
            dist.all_reduce(n)
        raw.train()
        return (tot / n).item()

    best = float("inf")
    hist = []

    def log_val(step, epoch):
        nonlocal best
        vl = evaluate()
        best = min(best, vl)
        if master:
            print(f"VAL step {step} epoch {epoch:.2f} heldout_resp_nll {vl:.4f} min {best:.4f}", flush=True)
            hist.append({"step": step, "epoch": epoch, "val_nll": vl, "val_nll_min": best})
            with open(os.path.join(out, "val.jsonl"), "a") as fh:
                fh.write(json.dumps(hist[-1]) + "\n")
            if wb:
                wb.log({"step": step, "epoch": epoch, "val/resp_nll": vl, "val/resp_nll_min": best})

    log_val(0, 0.0)
    raw.train()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        for bidx in epoch_plan(len(ex), gb, args.seed, ep, lens):
            # zipper-split the length-sorted global batch so ranks see similar lengths
            if len(bidx) < world:  # every rank needs at least one example (DDP sync)
                continue
            gbatch = [ex[i] for i in sorted(bidx, key=lambda i: lens[i])]
            n_resp = sum(len(t) - s_ for t, s_ in gbatch)
            mbs = micro_batches(gbatch[rank::world], args.tok_budget, args.pad_to)
            apply_schedule(mu, adam, step, total, warm, decay_frac=1.0, mom_warmup=1, mom0=0.95, mom1=0.95)
            lsum = torch.zeros((), device=dev)
            for i, mb in enumerate(mbs):
                x, y = make_xy(mb, args.pad_to, args.tok_budget)
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                k = (y != -1).sum()
                last = i == len(mbs) - 1
                ctx = model.no_sync() if (ddp and not last) else torch.enable_grad()
                with ctx, torch.autocast("cuda", dtype=amp):
                    loss = model(x, y) * k / n_resp * world  # DDP averages grads over ranks
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
            if master and (step % 10 == 0 or step == 1):
                el = time.time() - t0
                print(f"step {step}/{total} ep {ep} loss {lsum.item():.4f} lr_f "
                      f"{adam.param_groups[0]['lr'] / adam.param_groups[0]['base_lr']:.3f} "
                      f"{el / step:.2f}s/step scale {scaler.get_scale():.0f}", flush=True)
                if wb:
                    wb.log({"step": step, "epoch": step / steps_per_epoch, "train/loss": lsum.item()})
            if step % args.val_every == 0:
                log_val(step, step / steps_per_epoch)
        if step % args.val_every:
            log_val(step, step / steps_per_epoch)
        if master:
            p = os.path.join(out, f"epoch{ep + 1}.pt")
            torch.save({"model": raw.state_dict(), "cfg": cfg.to_dict(), "step": step, "epoch": ep + 1,
                        "init": os.path.abspath(args.init), "val_nll": hist[-1]["val_nll"]}, p + ".tmp")
            os.replace(p + ".tmp", p)
            print(f"saved {p}", flush=True)
    if master:
        res = {"name": args.name, "final_val_nll": hist[-1]["val_nll"], "min_val_nll": best,
               "init_val_nll": hist[0]["val_nll"], "steps": total, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
        print("RESULT", json.dumps(res), flush=True)
        if wb:
            wb.summary.update(res)
            wb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
