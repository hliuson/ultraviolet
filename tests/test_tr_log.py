"""Key mapping of scripts/tr_log.py on small fixtures cut from real outputs (tests/fixtures/tr_log/):
eval_rl_ckpt main/passk and an rl_grpo log.jsonl eval + train record."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import tr_log as T  # noqa: E402

FX = os.path.join(HERE, "fixtures", "tr_log")


def fx(name):
    return json.load(open(os.path.join(FX, name)))


def test_math_keys_eval_rl_ckpt():
    out = T.math_keys(fx("eval_rl_ckpt_main.json"))
    assert "sum/math_think_greedy" not in out  # greedy is not a headline key; it stays in diag/math
    assert out["diag/math/a_normal/greedy/gr_big/acc"] == pytest.approx(0.066)
    assert out["sum/math_think_t07"] == pytest.approx(0.059)
    assert out["sum/math_nothink_t07"] == pytest.approx(0.04625)
    assert out["sum/math_ao_t07"] == pytest.approx(0.003)
    assert out["sum/math_gap_ab"] == pytest.approx(0.059 - 0.04625)
    assert out["sum/close_rate"] == pytest.approx(0.9715)
    assert out["sum/think_len"] == pytest.approx(554.035)
    assert out["sum/nomm_think_t07"] == pytest.approx(0.09561128526645768)
    assert out["diag/math/a_normal/t0.7x4/gr_big/pass@4"] > 0
    assert not any(k.startswith(("task/", "train/")) for k in out)


def test_math_keys_same_from_rl_log_record():
    rec = json.loads(open(os.path.join(FX, "rl_log.jsonl")).readline())
    out = T.math_keys(rec)  # {"step", "eval"} wrapper
    assert out == T.math_keys(rec["eval"])
    assert out["sum/math_think_t07"] == pytest.approx(rec["eval"]["a_normal/t0.7x4/gr_big"]["acc"])
    assert out["sum/math_gap_ab"] == pytest.approx(rec["eval"]["a_normal/t0.7x4/gr_big"]["acc"]
                                                   - rec["eval"]["b_nothink/t0.7x4/gr_big"]["acc"])


def test_passk_and_coverage():
    p = fx("eval_rl_ckpt_passk.json")
    out = T.passk_keys(p)
    assert out["sum/pass1"] == pytest.approx(0.05040625)
    assert out["sum/pass64"] == pytest.approx(0.344)
    # coverage buckets on a toy (P=4, K=64) array
    init = np.zeros((4, 64), bool)
    run = np.zeros((4, 64), bool)
    init[0, :2] = True; run[0, :10] = True   # sharpened  # noqa: E702
    run[1, :1] = True                          # new
    init[2, :5] = True                         # lost
    init[3, :5] = True; run[3, :5] = True    # unchanged  # noqa: E702
    out = T.passk_keys(p, run, init)
    assert (out["sum/sharpened"], out["sum/new"], out["sum/lost"], out["sum/unchanged"]) == (1, 1, 1, 1)
    assert out["sum/pass1"] == pytest.approx(run.mean())
    assert out["sum/pass64"] == pytest.approx(0.75)
    assert 0 <= out["sum/null_new32"] <= 4 and "sum/new32" in out


def test_rl_train_keys():
    rec = [json.loads(line) for line in open(os.path.join(FX, "rl_log.jsonl"))][1]
    out = T.rl_train_keys(rec)
    assert out["train/reward"] == rec["reward"]
    assert out["train/signal_frac"] == rec["frac_groups_nonzero_adv"]
    assert out["train/rollouts_total"] == rec["rollouts_total"]
    assert out["diag/tis_mean"] == rec["tis_mean"]
    assert "step" not in out and "train/step" not in out


def test_val_keys():
    assert T.pt_val_keys({"step": 5, "bpb": 0.5, "bpb_v1": 0.7, "tokens": 9}) == {
        "sum/val_bpb": 0.5, "sum/val_bpb_v1": 0.7, "train/tokens": 9}
    assert T.sft_val_keys({"step": 1, "epoch": 0.5, "val_nll": 0.9, "val_nll_min": 0.8})["sum/heldout_nll"] == 0.9


def test_math_keys_t1_headline():
    r = {"a_normal/t1.0x16/gr_big": {"acc": 0.2, "pass@16": 0.5, "maj@16": 0.3, "close_rate": 0.9, "mean_len": 500.0},
         "b_nothink/t1.0x16/gr_big": {"acc": 0.05, "maj@16": 0.06},
         "c_answer_only/t1.0x16/gr_big": {"acc": 0.02, "maj@16": 0.03}}
    out = T.math_keys(r)
    assert out["sum/math_pass1"] == 0.2 and out["sum/math_maj16"] == 0.3 and out["sum/math_pass16_think"] == 0.5
    assert out["sum/math_nothink_pass1"] == 0.05 and out["sum/math_ao_pass1"] == 0.02
    assert out["sum/math_gap_ab"] == pytest.approx(0.15) and out["sum/close_rate"] == 0.9
    pk = T.passk_keys({"passk": {"K": 64, "temp": 1.0, "cond": "think", "pass@1": 0.21, "pass@64": 0.55}})
    assert pk["sum/math_pass1"] == 0.21 and pk["sum/pass64"] == 0.55
    assert "sum/math_pass1" not in T.passk_keys({"passk": {"K": 64, "temp": 1.0, "cond": "answer", "pass@1": 0.1}})
