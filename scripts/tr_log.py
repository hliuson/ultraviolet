"""Shared wandb logger for pretrain -> SFT/midtrain -> GRPO RL runs (README.md, wandb).

Schema:
  sum/   headline held-out math on gr_big2 (1000 guess-resistant problems): math_think_greedy, math_think_t07,
         math_nothink_t07, math_ao_t07, math_gap_ab (think - nothink at T=0.7), pass1/pass8/pass64 (T=1.0),
         close_rate, think_len, nomm_think_t07, coverage vs init (sharpened/new/lost/unchanged,
         new32/lost32 and the init-vs-init null_new32/null_lost32); pretrain val_bpb / val_bpb_v1; heldout_nll.
  train/ reward, mean_len, close_rate, signal_frac, rollouts_total, hours, lr_f, grad_norm; loss for SFT/pretrain.
  diag/  everything else.

Use from a script:
    import tr_log
    tr_log.init("rl-foo", "rl", parent="mid-alpha", data="pool_v1/pool_gr_train", mix={...}, config=vars(args))
    tr_log.log(tr_log.rl_train_keys(rec), step=k)
    tr_log.log_math(run_eval_result, step=k)          # eval_rl_ckpt / rl_grpo run_eval dicts
    tr_log.finish()
The *_keys functions are pure (dict -> flat dict) and are what tests/test_tr_log.py checks.
"""
import json
import math
import os
import re

import numpy as np

PROJECT = os.environ.get("WANDB_PROJECT", "ultraviolet")
ENTITY = os.environ.get("WANDB_ENTITY") or None
STAGES = ("pretrain", "sft", "mid", "rl", "eval")

_state = {"run": None, "last": None}


# ----------------------------------------------------------------------------------------------- run management
def init(name, stage, parent=None, data=None, mix=None, config=None, tags=None, key=False, detail_run=None,
         ckpt_path=None, notes=None, run_id=None, dir=None, mode=None):
    """Open (or resume) the run with id = run_id or name. Re-running is idempotent: steps <= the last step already
    in the run are skipped by log(). Wandb never reuses a deleted id, so for a full rewrite pass run_id=name+'-r2'."""
    import wandb
    assert stage in STAGES, stage
    cfg = dict(config or {})
    cfg["stage"] = stage
    for k, v in (("parent", parent), ("data", data), ("mix", mix), ("detail_run", detail_run),
                 ("ckpt_path", ckpt_path)):
        if v is not None:  # never blank out a field another writer set (e.g. a summary-only update)
            cfg[k] = v
    tg = set([stage] + list(tags or []) + (["key"] if key else []))
    d = dir or os.environ.get("WANDB_DIR") or os.getcwd()
    os.makedirs(d, exist_ok=True)
    run = wandb.init(project=PROJECT, entity=ENTITY, id=run_id or name, name=name, resume="allow",
                     notes=notes, dir=d, mode=mode, settings=wandb.Settings(silent=True, init_timeout=120))
    run.config.update(cfg, allow_val_change=True)
    run.tags = tuple(sorted(tg | set(run.tags or ())))
    last = run.summary.get("_step")
    _state.update(run=run, last=None if last is None else int(last))
    return run


def last_step():
    """Last step already in the (resumed) run, or None for a new run."""
    return _state["last"]


def log(d, step=None):
    """Log a flat dict at `step`. Steps at or below the resumed run's last step are skipped (idempotent backfill).
    Logging the same step twice in one process merges the rows (wandb commits a row when a higher step arrives)."""
    run = _state["run"]
    if run is None or not d:
        return False
    d = {k: v for k, v in d.items() if _ok(v)}
    if step is None:
        run.log(d)
        return True
    if _state["last"] is not None and step <= _state["last"]:
        return False
    run.log(d, step=int(step))
    return True


def summary(d):
    run = _state["run"]
    if run is not None:
        run.summary.update({k: v for k, v in d.items() if _ok(v)})


def finish():
    run = _state["run"]
    if run is not None:
        run.finish()
    _state.update(run=None, last=None)


def _ok(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, np.integer, np.floating)):
        return math.isfinite(float(v))
    return isinstance(v, str)


# ----------------------------------------------------------------------------------------------- math (gr_big2)
def _unwrap_res(res):
    """Accept eval_rl_ckpt output {"ckpt", "res"}, an rl_grpo log.jsonl eval record {"step", "eval"}, or the bare
    run_eval dict."""
    if isinstance(res, dict) and "res" in res and isinstance(res["res"], dict):
        return res["res"]
    if isinstance(res, dict) and "eval" in res and isinstance(res["eval"], dict):
        return res["eval"]
    return res


def _t_tag(res, temp="0.7"):
    pat = re.compile(r"^a_normal/(t" + re.escape(temp) + r"x\d+)/")
    for k in res:
        m = pat.match(k)
        if m:
            return m.group(1)
    return None


def math_keys(res, diag_subsets=("gr_big", "nomm")):
    """run_eval / eval_rl_ckpt result -> sum/ (gr_big2 headline) + diag/math/ keys. Includes passk if present."""
    r = _unwrap_res(res)
    out = {}
    g = lambda k, m="acc": (r.get(k) or {}).get(m)  # noqa: E731
    t = _t_tag(r)
    # headline (no greedy): T=1.0 sampling -- sum/math_pass1 (mean over k samples), sum/math_maj16,
    # and sum/pass64 from the final passk run. Greedy rows, if any, only go to diag/math/.
    t1 = _t_tag(r, "1.0")
    if t1:
        a1, b1, c1 = g(f"a_normal/{t1}/gr_big"), g(f"b_nothink/{t1}/gr_big"), g(f"c_answer_only/{t1}/gr_big")
        k1 = int(t1.split("x")[1])
        out["sum/math_pass1"] = a1
        out["sum/math_nothink_pass1"] = b1
        out["sum/math_ao_pass1"] = c1
        if a1 is not None and b1 is not None:
            out["sum/math_gap_ab"] = a1 - b1
        out[f"sum/math_maj{k1}"] = g(f"a_normal/{t1}/gr_big", f"maj@{k1}")
        out[f"sum/math_pass{k1}_think"] = g(f"a_normal/{t1}/gr_big", f"pass@{k1}")
        out["sum/close_rate"] = g(f"a_normal/{t1}/gr_big", "close_rate")
        out["sum/think_len"] = g(f"a_normal/{t1}/gr_big", "mean_len")
        out["sum/nomm_pass1"] = g(f"a_normal/{t1}/nomm")
        for c_ in ("b_nothink", "c_answer_only"):
            out[f"diag/math/{c_}_maj{k1}"] = g(f"{c_}/{t1}/gr_big", f"maj@{k1}")
    if t:
        a, b, c = g(f"a_normal/{t}/gr_big"), g(f"b_nothink/{t}/gr_big"), g(f"c_answer_only/{t}/gr_big")
        if a is not None:
            out["sum/math_think_t07"] = a
            if not t1:
                out["sum/close_rate"] = g(f"a_normal/{t}/gr_big", "close_rate")
                out["sum/think_len"] = g(f"a_normal/{t}/gr_big", "mean_len")
        if b is not None:
            out["sum/math_nothink_t07"] = b
        if c is not None:
            out["sum/math_ao_t07"] = c
        if a is not None and b is not None and not t1:
            out["sum/math_gap_ab"] = a - b
        if g(f"a_normal/{t}/nomm") is not None:
            out["sum/nomm_think_t07"] = g(f"a_normal/{t}/nomm")
    for k, d in r.items():
        if not isinstance(d, dict):
            continue
        parts = k.split("/")
        if k.startswith("e_forced_close/"):
            for m, v in d.items():
                out[f"diag/math/{k}/{m}"] = v
        elif len(parts) == 3 and parts[2] in diag_subsets:
            for m, v in d.items():
                out[f"diag/math/{k}/{m}"] = v
    if "passk" in r:
        out.update(passk_keys(r["passk"]))
    return {k: v for k, v in out.items() if v is not None}


def log_math(res, step=None):
    return log(math_keys(res), step)


# ----------------------------------------------------------------------------------------------- pass@k + coverage
def pass_at(n, c, k):
    if n - c < k:
        return 1.0
    q = 1.0
    for i in range(k):
        q *= (n - c - i) / (n - i)
    return 1.0 - q


def coverage_keys(ok, init_ok, draws=20, half=32, seed=0):
    """Coverage buckets vs init: sharpened (init solves >=1, run rate higher), new (init 0/K, run
    >=1), lost (init >=1, run 0/K), unchanged; plus the 32-vs-32 new/lost and the init-vs-init null (20 draws)."""
    ok, init_ok = np.asarray(ok, bool), np.asarray(init_ok, bool)
    assert ok.shape[0] == init_ok.shape[0], (ok.shape, init_ok.shape)
    ri, rr = init_ok.mean(1), ok.mean(1)
    si, sr = init_ok.any(1), ok.any(1)
    out = {"sum/sharpened": int((si & (rr > ri)).sum()), "sum/new": int((~si & sr).sum()),
           "sum/lost": int((si & ~sr).sum())}
    out["sum/unchanged"] = len(ri) - out["sum/sharpened"] - out["sum/new"] - out["sum/lost"]
    out["diag/coverage/init_solvable"] = int(si.sum())
    if si.any():
        out["diag/coverage/drate_init_solvable"] = float((rr - ri)[si].mean())
    Ki, Kr = init_ok.shape[1], ok.shape[1]
    if Ki >= 2 * half and Kr >= half:
        rng = np.random.RandomState(seed)
        nw, ls, nn, nl = [], [], [], []
        for _ in range(draws):
            pi = rng.permutation(Ki)
            a, b = init_ok[:, pi[:half]].any(1), init_ok[:, pi[half:2 * half]].any(1)
            r_ = ok[:, rng.permutation(Kr)[:half]].any(1)
            nw.append((~a & r_).sum())
            ls.append((a & ~r_).sum())
            nn.append((~a & b).sum())
            nl.append((a & ~b).sum())
        out.update({"sum/new32": float(np.mean(nw)), "sum/lost32": float(np.mean(ls)),
                    "sum/null_new32": float(np.mean(nn)), "sum/null_lost32": float(np.mean(nl))})
    return out


def passk_keys(passk_json, npy_path=None, init_npy=None):
    """eval_rl_ckpt --passk output ({"ckpt","res":{"passk":...}} or the passk dict) [+ its (P,K) .npy and the init's
    .npy] -> sum/pass1, pass8, pass64 (T=1.0), coverage vs init, diag/passk/*."""
    p = passk_json
    if isinstance(p, str):
        p = json.load(open(p))
    p = _unwrap_res(p)
    p = p.get("passk", p)
    K = p.get("K")
    out = {}
    if "pass@1" in p:
        out["sum/pass1"] = p["pass@1"]
    if K and f"pass@{K}" in p:
        out[f"sum/pass{K}"] = p[f"pass@{K}"]
    for k in ("frac_forced", "mean_len", "temp", "K"):
        if k in p:
            out[f"diag/passk/{k}"] = p[k]
    if p.get("cond"):
        out["diag/passk/cond_answer_only"] = int(p["cond"] == "answer")
    ok = np.load(npy_path) if isinstance(npy_path, str) else npy_path
    if ok is not None:
        ok = np.asarray(ok, bool)
        Kn = ok.shape[1]
        cnt = ok.sum(1)
        for k in (1, 8, 64):
            if k <= Kn:
                out[f"sum/pass{k}"] = float(np.mean([pass_at(Kn, int(c), k) for c in cnt]))
        if init_npy is not None:
            init_ok = np.load(init_npy) if isinstance(init_npy, str) else init_npy
            out.update(coverage_keys(ok, init_ok))
    if out.get("sum/pass1") is not None and p.get("cond", "think") == "think" and float(p.get("temp", 1.0)) == 1.0:
        out["sum/math_pass1"] = out["sum/pass1"]  # T=1.0 think pass@1 over K samples: the headline (preferred over k=16)
    return out


# ----------------------------------------------------------------------------------------------- training records
RL_TRAIN = {"reward": "reward", "mean_len": "mean_len", "close_rate": "close_rate",
            "frac_groups_nonzero_adv": "signal_frac", "frac_groups_all_correct": "frac_all_correct",
            "rollouts_total": "rollouts_total", "hours": "hours", "lr_f": "lr_f", "grad_norm": "grad_norm",
            "overlong_frac": "overlong_frac",
            "overlong_pen_mean": "overlong_pen_mean", "frac_groups_len_only": "frac_groups_len_only",
            "reward_true": "reward_true"}


def rl_train_keys(rec, rollouts_total=None):
    """rl_grpo log.jsonl training record -> train/ (headline) + diag/ (everything else numeric)."""
    out = {}
    for k, v in rec.items():
        if k == "step" or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[f"train/{RL_TRAIN[k]}" if k in RL_TRAIN else f"diag/{k}"] = v
    if rollouts_total is not None and "train/rollouts_total" not in out:
        out["train/rollouts_total"] = rollouts_total
    return out


def sft_val_keys(rec):
    """sft.py val.jsonl row -> sum/heldout_nll (held-out trace NLL), diag/heldout_nll_min, train/epoch."""
    out = {"sum/heldout_nll": rec.get("val_nll"), "diag/heldout_nll_min": rec.get("val_nll_min"),
           "train/epoch": rec.get("epoch")}
    return {k: v for k, v in out.items() if v is not None}


def pt_val_keys(rec):
    """train_pt.py val.jsonl row -> sum/val_bpb, sum/val_bpb_v1, diag/val_loss*, train/tokens."""
    m = {"bpb": "sum/val_bpb", "bpb_v1": "sum/val_bpb_v1", "val_loss": "diag/val_loss",
         "val_loss_v1": "diag/val_loss_v1", "tokens": "train/tokens"}
    return {m[k]: v for k, v in rec.items() if k in m}
