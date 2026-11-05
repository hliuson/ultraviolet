"""Pretrain the math LM (ultraviolet.model) on uint16/uint32 token shards.

Single GPU:  python scripts/train_pt.py --data DIR --preset v-deep-mqa --tokens 2e8 --out RUN --name NAME
DDP:         torchrun --standalone --nproc_per_node 4 scripts/train_pt.py ...
Precision:   --dtype fp16 (Turing / 2080 Ti: autocast + GradScaler) or bf16 (Ampere+/Blackwell: autocast, no scaler).
Resumes from RUN/ckpt.pt automatically. wandb: project $WANDB_PROJECT (default ultraviolet), group math-pt, id mpt-<name> (stable).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from ultraviolet.data import TokenShards, TraceStream, trace_rows  # noqa: E402
from ultraviolet.model import MathConfig, MathLM, PRESETS, make_config  # noqa: E402
from ultraviolet.optim import apply_schedule, build_optimizers  # noqa: E402

PROJECT, GROUP = os.environ.get("WANDB_PROJECT", "ultraviolet"), "math-pt"


class TRPt:
    """wb.log/summary/finish calls routed to tr_log."""
    def __init__(self, args, config):
        import types
        import tr_log
        self.tl = tr_log
        self.val_name = args.val_name
        tr_log.init(args.tr_name, "pretrain", data=args.tr_data or None, ckpt_path=args.out, dir=args.out,
                    config={"args": vars(args), **config})
        self.summary = types.SimpleNamespace(update=lambda d: tr_log.summary(
            {f"diag/{k}": v for k, v in d.items() if isinstance(v, (int, float))}))

    def log(self, d):
        if "val/loss" in d:
            row = {"val_loss": d["val/loss"], "bpb": d["val/bpb"], "tokens": d["tokens"]}
            row.update({k[4:]: v for k, v in d.items() if k.startswith("val/val_loss_") or k.startswith("val/bpb_")})
            kv = self.tl.pt_val_keys(row)
            if self.val_name:  # --val-name / --mix extras / --heldout-traces (pretrain mix comparisons)
                kv[f"sum/val_bpb_{self.val_name}"] = d["val/bpb"]
                kv.update({f"sum/val_bpb_{k[8:]}": v for k, v in d.items() if k.startswith("val/bpb_")})
                kv.update({f"sum/heldout_nll_{k[12:]}": v for k, v in d.items() if k.startswith("val/heldout_")})
                kv.update({f"train/{k[4:]}": v for k, v in d.items() if k in ("val/hours", "val/train_hours")})
        else:
            kv = {"train/loss": d["train/loss"], "train/lr_f": d["lr_factor"], "train/tokens": d["tokens"]}
        self.tl.log(kv, step=d["step"])

    def finish(self):
        self.tl.finish()


def mix_owner(step, fracs, G, rank, world):
    """--mix: source index (-1 = --data) of each of this rank's G // world sequences at `step`. Source k gets
    floor-cumulative frac_k * G rows per step (exact on average, as trace_rows); rows are dealt round-robin to ranks."""
    cnt = [b - a for a, b in (trace_rows(step, w, G) for w in fracs)]
    own = np.concatenate([np.full(c, k) for k, c in enumerate(cnt)] + [np.full(G - sum(cnt), -1)])
    return own[rank::world]


def wandb_init(run_dir, name, config, mode):
    if mode == "disabled":
        return None
    import wandb
    kw = dict(project=PROJECT, group=GROUP, name=name, id=f"mpt-{name}", resume="allow", config=config,
              dir=run_dir)
    try:
        run = wandb.init(mode="online" if mode == "auto" else mode, settings=wandb.Settings(silent=True,
                         init_timeout=90), **kw)
    except Exception as e:  # noqa: BLE001
        if mode == "online":
            raise
        print(f"wandb online init failed ({type(e).__name__}); offline", flush=True)
        run = wandb.init(mode="offline", settings=wandb.Settings(silent=True), **kw)
    run.define_metric("tokens")
    run.define_metric("*", step_metric="tokens")
    return run


def parse():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--data", required=True)
    a("--out", required=True)
    a("--name", required=True)
    a("--preset", default="v-deep-mqa", choices=list(PRESETS))
    a("--cfg", default=None, help="JSON dict of MathConfig overrides")
    a("--xsa", type=int, default=None)
    a("--mudd", type=int, default=None)
    a("--seq-len", type=int, default=2048)
    a("--batch-tokens", type=int, default=262144)
    a("--micro-bs", type=int, default=16)
    a("--tokens", type=float, default=2e8)
    a("--extra-val", nargs="*", default=[], help="NAME=DATA_DIR: also report val bpb on DATA_DIR's val split "
      "(same tokenizer), e.g. v1=/path/cdata16k")
    a("--trace-data", default=None, help="TraceStream dir (tokens.npy/offsets.npy) mixed in at --trace-frac")
    a("--trace-frac", type=float, default=0.0, help="fraction of training sequences drawn from --trace-data")
    a("--mix", nargs="*", default=[], help="NAME=DATA_DIR:FRAC ...: draw FRAC of each step's training sequences "
      "from DATA_DIR's train split (same tokenizer); the rest come from --data. Not combinable with --trace-data")
    a("--heldout-traces", default=None, help="held-out trace jsonl (prep_traces.py): thinking/solution NLL at every val")
    a("--tok", default=None, help="tokenizer.json (for --heldout-traces)")
    a("--val-name", default="", help="tr_log: also log the main val bpb as sum/val_bpb_<NAME>")
    a("--stop-tokens", type=float, default=0, help="stop early (checkpoint + exit) after this many tokens; the LR "
      "schedule still follows --tokens, so truncated arms are comparable at matched tokens")
    a("--warmup", type=int, default=100)
    a("--decay-frac", type=float, default=0.2)
    a("--final-lr-frac", type=float, default=0.0, help="decay ends at this fraction of peak LR (MAI-Base-1: 0.1)")
    a("--muon-lr", type=float, default=0.018)
    a("--lr-mult", type=float, default=1.0)
    a("--adam-lr-mult", type=float, default=1.0)
    a("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    a("--compile", type=int, default=0, help="1: compile the trunk; 2: compile the whole training forward incl. loss")
    a("--attn", default="sdpa", choices=["sdpa", "flex", "varlen"], help="training attention: dense-mask SDPA, "
      "FlexAttention, or the CUTLASS varlen kernel with window_size (model.train_attn = 'varlen', docs/perf_opt.md)")
    a("--doc-mask", type=int, default=0, help="1: windows start at a document start (TokenShards.sample_docs), "
      "attention, RoPE positions and the bigram reset within each document (EOS = BOS of the next doc); docs longer "
      "than --max-doc-len split into fresh segments. Runs through the varlen kernel (also for evals)")
    a("--max-doc-len", type=int, default=2048, help="--doc-mask: cap on one segment (0: no cap, up to --seq-len)")
    a("--bigram-rows", type=int, default=None, help="hashed bigram embedding rows (MathConfig.bigram_rows; 0 = off)")
    a("--bigram-dim", type=int, default=None, help="bigram table width (MathConfig.bigram_dim; 0 = d_model)")
    a("--ve-block-pos", default=None, help="comma list: value embeddings on layers l %% global_every in this set "
      "(MathConfig.ve_block_pos), e.g. 1,4 for uv-50m-alpha")
    a("--val-every", type=int, default=100)
    a("--val-tokens", type=float, default=2e6)
    a("--log-every", type=int, default=10)
    a("--ckpt-every", type=int, default=500)
    a("--keep-every-tokens", type=float, default=0, help="also keep a copy RUN/stable_<X>B.pt each time training "
      "crosses a multiple of this many tokens (WSD branch points)")
    a("--seed", type=int, default=0)
    a("--wandb-mode", default="auto", choices=["auto", "online", "offline", "disabled"])
    a("--tr-name", default="", help="log through tr_log under this id instead of a plain wandb run")
    a("--tr-data", default="", help="tr_log data id (corpus)")
    return ap.parse_args()


def main():
    args = parse()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, lrank = 0, 1, 0
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    train = TokenShards(args.data, "train")
    val = TokenShards(args.data, "val")
    over = json.loads(args.cfg) if args.cfg else {}
    over["vocab_size"] = over.get("vocab_size", train.meta["vocab"])
    if args.xsa is not None:
        over["xsa"] = bool(args.xsa)
    if args.mudd is not None:
        over["mudd"] = bool(args.mudd)
    if args.bigram_rows is not None:
        over["bigram_rows"] = args.bigram_rows
    if args.bigram_dim is not None:
        over["bigram_dim"] = args.bigram_dim
    if args.ve_block_pos is not None:
        over["ve_block_pos"] = [int(x) for x in args.ve_block_pos.split(",") if x != ""]
    cfg = make_config(args.preset, **over)

    torch.manual_seed(args.seed)
    model = MathLM(cfg).to(dev)
    if args.attn == "flex" and torch.cuda.get_device_capability(dev)[0] < 8:
        print("flex attention: pre-Ampere GPU (64KB smem) -> using sdpa", flush=True)
        args.attn = "sdpa"
    if args.doc_mask:
        assert args.attn != "flex", "--doc-mask runs through the varlen kernel; use --attn sdpa|varlen"
    if args.attn == "varlen":
        model.train_attn = "varlen"
    else:
        model.attn_impl = args.attn
    raw = model
    eos_id = int(train.meta.get("eos_id", 0))

    def docs_of(x):  # --doc-mask inputs for one (b, T) micro-batch (computed outside any compiled region)
        return raw.doc_inputs(x, eos_id, args.max_doc_len) if args.doc_mask else None

    def draw(src, n, r_):
        return src.sample_docs(n, T, r_, args.max_doc_len) if args.doc_mask else src.sample(n, T, r_)
    T = args.seq_len
    per_step = world * args.micro_bs * T
    accum = max(1, args.batch_tokens // per_step)
    step_tokens = per_step * accum
    total = int(math.ceil(args.tokens / step_tokens))
    mu, adam = build_optimizers(raw, muon_lr=args.muon_lr, lr_mult=args.lr_mult, adam_lr_mult=args.adam_lr_mult,
                                distributed=ddp)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16", init_scale=2.0 ** 14)

    start = 0
    ck = os.path.join(args.out, "ckpt.pt")
    best = float("inf")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        raw.load_state_dict(s["model"])
        mu.load_state_dict(s["mu"])
        adam.load_state_dict(s["adam"])
        scaler.load_state_dict(s["scaler"])
        start, best = s["step"], s.get("best_val", best)
        if master:
            print(f"resumed from step {start}", flush=True)

    if args.compile:  # evals call raw, so its trunk is compiled in both modes (eager flex is very slow)
        raw.trunk = torch.compile(raw.trunk)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[lrank])
    if args.compile == 2:  # training forward incl. loss
        model = torch.compile(model)

    pc = cfg.param_counts()
    run_cfg = {"args": vars(args), "model": cfg.to_dict(), "params": pc, "world": world, "accum": accum,
               "step_tokens": step_tokens, "total_steps": total, "data_meta": {k: train.meta.get(k) for k in
                                                                               ("vocab", "train", "val")}}
    if master:
        print(json.dumps(run_cfg, default=str), flush=True)
        json.dump(run_cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1, default=str)
    wb = None
    if master and args.tr_name and args.wandb_mode != "disabled":
        wb = TRPt(args, run_cfg)
    elif master:
        wb = wandb_init(args.out, args.name, run_cfg, args.wandb_mode)

    vwin = val.fixed_windows(int(args.val_tokens), T)
    vwin = vwin[rank::world]
    bpt = val.meta["val"]["bytes_per_token"]
    extra = []
    for spec in args.extra_val:
        nm, d = spec.split("=", 1)
        ev = TokenShards(d, "val")
        assert ev.meta["vocab"] == val.meta["vocab"], f"{nm}: tokenizer mismatch"
        extra.append((nm, ev.fixed_windows(int(args.val_tokens), T)[rank::world], ev.meta["val"]["bytes_per_token"]))

    mixes = []
    for spec in args.mix:
        nm, rest = spec.split("=", 1)
        d, w = rest.rsplit(":", 1)
        ms = TokenShards(d, "train")
        assert ms.meta["vocab"] == train.meta["vocab"], f"{nm}: tokenizer mismatch"
        mixes.append((nm, ms, float(w)))
    assert not (mixes and args.trace_data), "--mix and --trace-data are exclusive"
    assert sum(w for _, _, w in mixes) <= 1
    hrecs = [json.loads(l) for l in open(args.heldout_traces)] if args.heldout_traces else []
    if hrecs:
        from tokenizers import Tokenizer
        sys.path.insert(0, HERE)
        from eval_math import trace_nll
        htok = Tokenizer.from_file(args.tok)
    t_start, t_eval = time.time(), 0.0

    def heldout():  # master only; fixed (micro_bs, T) shapes so the compiled trunk does not recompile
        def padded(x):
            xp = torch.zeros((args.micro_bs, T), dtype=x.dtype, device=x.device)
            xp[:x.shape[0], :x.shape[1]] = x
            return raw(xp)[:x.shape[0], :x.shape[1]]
        raw.eval()
        with torch.no_grad():
            r = trace_nll(padded, htok, hrecs, T, args.micro_bs)
        raw.train()
        return {"heldout_think": r["thinking"], "heldout_sol": r["solution"]}

    def evaluate(vwin=vwin):
        raw.eval()
        tot, n = torch.zeros((), device=dev), torch.zeros((), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=args.dtype != "fp32"):
            for i in range(0, len(vwin), args.micro_bs):
                xb = vwin[i:i + args.micro_bs].to(dev, non_blocking=True)
                loss = raw(xb[:, :-1], xb[:, 1:], docs=docs_of(xb[:, :-1]))
                k = xb[:, 1:].numel()
                tot += loss.float() * k
                n += k
        if ddp:
            dist.all_reduce(tot)
            dist.all_reduce(n)
        raw.train()
        return (tot / n).item()

    def save(step, path):
        torch.save({"model": raw.state_dict(), "mu": mu.state_dict(), "adam": adam.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "cfg": cfg.to_dict(), "best_val": best,
                    "args": vars(args)}, path + ".tmp")
        os.replace(path + ".tmp", path)

    model.train()
    t_last, tok_last = time.time(), 0
    skipped = 0
    loss_acc = torch.zeros((), device=dev)
    # background loader: batches are a pure function of (seed, rank, step), so prefetching keeps runs reproducible
    import queue
    import threading
    q = queue.Queue(maxsize=4)

    tstream = TraceStream(args.trace_data, args.seed) if args.trace_data and args.trace_frac > 0 else None
    G = step_tokens // T  # global sequences per step (step_tokens is already global)
    if tstream is not None and master:
        drawn = args.trace_frac * total * step_tokens
        rep_info = {"trace_unique_tokens": tstream.L, "trace_docs": tstream.n, "trace_frac": args.trace_frac,
                    "trace_tokens_drawn": int(drawn), "trace_repetition": drawn / tstream.L}
        print("TRACE MIX", json.dumps(rep_info), flush=True)
        json.dump(rep_info, open(os.path.join(args.out, "trace_mix.json"), "w"), indent=1)

    def producer():
        for s_ in range(start, total):
            r_ = np.random.default_rng([args.seed, rank, s_])
            nb = args.micro_bs * accum
            if mixes:
                own = mix_owner(s_, [w for _, _, w in mixes], G, rank, world)
                parts = [draw(ms, int((own == k).sum()), r_) for k, (_, ms, _) in enumerate(mixes)]
                parts.append(draw(train, int((own == -1).sum()), r_))
                bt = torch.cat(parts)
                q.put(bt[torch.from_numpy(r_.permutation(nb))].pin_memory())
                continue
            if tstream is None:
                q.put(draw(train, nb, r_).pin_memory())
                continue
            a_, b_ = trace_rows(s_, args.trace_frac, G)
            mine = [j for j in range(a_, b_) if j % world == rank][:nb]
            rows = [torch.from_numpy(tstream.window(j, T)) for j in mine]
            if nb - len(mine):
                rows.append(draw(train, nb - len(mine), r_))
            bt = torch.cat([r.view(-1, T + 1) for r in rows])
            q.put(bt[torch.from_numpy(r_.permutation(nb))].pin_memory())

    threading.Thread(target=producer, daemon=True).start()
    for step in range(start, total):
        f = apply_schedule(mu, adam, step, total, args.warmup, args.decay_frac, final=args.final_lr_frac)
        batch = q.get().to(dev, non_blocking=True)
        loss_acc.zero_()
        for i in range(accum):
            xb = batch[i * args.micro_bs:(i + 1) * args.micro_bs]
            sync = (i == accum - 1) or not ddp
            ctx = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=amp_dtype, enabled=args.dtype != "fp32"):
                loss = (model(xb[:, :-1], xb[:, 1:]) if not args.doc_mask else
                        model(xb[:, :-1], xb[:, 1:], docs=docs_of(xb[:, :-1]))) / accum
                scaler.scale(loss).backward()
            loss_acc += loss.detach()
        scaler.unscale_(mu)
        scaler.unscale_(adam)
        sc0 = scaler.get_scale()
        scaler.step(mu)
        scaler.step(adam)
        scaler.update()
        if args.dtype == "fp16" and scaler.get_scale() < sc0:
            skipped += 1
        mu.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        tokens = (step + 1) * step_tokens
        if master and (step % args.log_every == 0 or step == total - 1):
            if ddp:
                pass
            l = loss_acc.item()
            now = time.time()
            tps = (tokens - tok_last) / max(1e-9, now - t_last) if tok_last or step > start else 0.0
            t_last, tok_last = now, tokens
            print(f"step {step}/{total} loss {l:.4f} lr_f {f:.3f} tok/s {tps:,.0f} scale {scaler.get_scale():.0f} "
                  f"skipped {skipped} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            if wb:
                wb.log({"tokens": tokens, "step": step, "train/loss": l, "lr_factor": f, "tok_per_s": tps,
                        "grad_scale": scaler.get_scale(), "skipped_steps": skipped})
        if (step + 1) % args.val_every == 0 or step == total - 1 or step == start:
            t_e0 = time.time()
            vl = evaluate()
            best = min(best, vl)
            ex = {}
            for nm, ew, ebpt in extra:
                el = evaluate(ew)
                ex[f"val_loss_{nm}"], ex[f"bpb_{nm}"] = el, el / math.log(2) / ebpt
            if hrecs and master:
                ex.update(heldout())
            if mixes or hrecs:  # wall hours since launch, and excluding eval time (train_hours)
                t_eval += time.time() - t_e0
                ex["hours"] = (time.time() - t_start) / 3600
                ex["train_hours"] = (time.time() - t_start - t_eval) / 3600
            if master:
                bpb = vl / math.log(2) / bpt
                print(f"VAL step {step} tokens {tokens} val_loss {vl:.4f} bpb {bpb:.4f} best {best:.4f} "
                      + " ".join(f"{k} {v:.4f}" for k, v in ex.items()), flush=True)
                if wb:
                    wb.log({"tokens": tokens, "step": step, "val/loss": vl, "val/bpb": bpb, "val/loss_min": best,
                            **{f"val/{k}": v for k, v in ex.items()}})
                with open(os.path.join(args.out, "val.jsonl"), "a") as fh:
                    fh.write(json.dumps({"step": step, "tokens": tokens, "val_loss": vl, "bpb": bpb, **ex}) + "\n")
        stop = args.stop_tokens and tokens >= args.stop_tokens
        if master and ((step + 1) % args.ckpt_every == 0 or step == total - 1 or stop):
            save(step + 1, ck)
        K = int(args.keep_every_tokens)
        if master and K and tokens // K > (tokens - step_tokens) // K and step < total - 1:
            save(step + 1, os.path.join(args.out, f"stable_{tokens / 1e9:.2f}B.pt"))
        if stop:
            if master:
                print(f"STOP at step {step + 1} tokens {tokens} (--stop-tokens)", flush=True)
            if wb:
                wb.finish()
            if ddp:
                dist.destroy_process_group()
            return
    if master:
        save(total, os.path.join(args.out, "final.pt"))
        res = {"name": args.name, "final_val_loss": vl, "best_val_loss": best, "final_bpb": vl / math.log(2) / bpt, **ex,
               "tokens": total * step_tokens, "params": pc}
        json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
        print("RESULT", json.dumps(res), flush=True)
        if wb:
            wb.summary.update(res)
            wb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
