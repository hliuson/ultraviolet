"""--mix row allocation in scripts/train_pt.py (mix_owner)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_pt import mix_owner  # noqa: E402


def test_mix_owner_shares_and_ranks():
    G, world, steps = 64, 2, 400
    fracs = [0.1, 0.25]
    tot = np.zeros(3)
    for s in range(steps):
        per = [mix_owner(s, fracs, G, r, world) for r in range(world)]
        assert all(len(p) == G // world for p in per)
        allr = np.concatenate(per)
        for k in (-1, 0, 1):
            tot[k + 1] += (allr == k).sum()
    got = tot / (G * steps)
    assert abs(got[1] - 0.1) < 1e-3 and abs(got[2] - 0.25) < 1e-3 and abs(got[0] - 0.65) < 1e-3


def test_mix_owner_zero_and_single_rank():
    assert (mix_owner(5, [0.0], 32, 0, 1) == -1).all()
    o = mix_owner(0, [0.5, 0.5], 32, 0, 1)
    assert (o == 0).sum() == 16 and (o == 1).sum() == 16
