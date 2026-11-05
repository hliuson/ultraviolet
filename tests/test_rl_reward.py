import os
import sys

import pytest

pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rl_grpo  # noqa: E402


class Tok:
    """o is already a str 'token list'; close id is the literal </think> marker in it."""

    def decode(self, o, skip_special_tokens=False):
        return o


C = "</think>"


def test_extract_direct():
    assert rl_grpo.extract_direct("think</think>\n\n42.\nmore") == "42"
    assert rl_grpo.extract_direct("t</think>\n\nThe answer is $\\boxed{7}$") == "7"
    assert rl_grpo.extract_direct("no close 5") is None
    assert rl_grpo.extract_direct("t</think>\n\n  ") is None


def test_answer_suffix():
    assert rl_grpo.answer_suffix(False, True) == ""
    assert rl_grpo.answer_suffix(True, False) == rl_grpo.ANS
    assert rl_grpo.answer_suffix(True, True) == "</think>\n\n"


def test_math_reward_requires_close():
    tk, it = Tok(), {"ref": "12"}
    assert rl_grpo.reward(tk, C, "x \\boxed{12}", it) == (0.0, False)
    assert rl_grpo.reward(tk, C, "x</think>\n\n\\boxed{12}", it) == (1.0, True)
    assert rl_grpo.reward(tk, C, "x</think>\n\n12", it, answer_direct=True) == (1.0, True)
    # answer-only: completion continues the forced suffix
    assert rl_grpo.reward(tk, C, "12}$", it, answer_only=True) == (1.0, True)
    assert rl_grpo.reward(tk, C, "12", it, answer_only=True, answer_direct=True) == (1.0, True)
    assert rl_grpo.reward(tk, C, "x</think>\n\n\\boxed{12}", {"answer": "12"}) == (1.0, True)  # eval-format ref


def test_task_reward_dispatch(monkeypatch):
    seen = {}

    def fake(item, text):
        seen["text"] = text
        return 1.0

    monkeypatch.setattr(rl_grpo, "task_score", fake)
    it = {"task": "gsm8k", "answer": "3"}
    assert rl_grpo.reward(Tok(), C, "no close", it) == (0.0, False)  # close still required
    assert rl_grpo.reward(Tok(), C, "a</think>b", it) == (1.0, True) and seen["text"] == "a</think>b"
    assert rl_grpo.reward(Tok(), C, "3", it, answer_only=True, answer_direct=True) == (1.0, True)
    assert seen["text"] == "</think>\n\n3"
