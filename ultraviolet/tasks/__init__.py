"""Verifiable task suite (reasoning-gym subset, GSM8K, MATH L1-2). See docs/tasks.md.

    from ultraviolet.tasks import score
    r = score(item["task"], completion_text, item)   # 1.0 / 0.0 (binary) or [0, 1] (graded)
"""
from ultraviolet.tasks.score import DROPPED_TASKS, TASKS, extract_answer, score, reward_kind, score_item, score_pred  # noqa: F401
