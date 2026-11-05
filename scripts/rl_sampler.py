"""Prompt samplers for rl_grpo.py.

DifficultyPoolSampler (after prime-rl's DifficultyPoolSampler): each problem's latest group-mean reward puts it in a
pool; prompts are drawn with replacement in proportion to the pool weight. Pools (default thresholds lo=0.25,
hi=0.75): unseen | zero (r == 0) | low (0, lo] | mid (lo, hi] | high (hi, 1) | solved (r == 1), default weights
1.0 / 0.1 / 0.5 / 1.0 / 0.3 / 0.05.

UniformSampler: epoch-wise permutation without replacement (rl_grpo's default).
"""
import numpy as np

POOLS = ("unseen", "zero", "low", "mid", "high", "solved")
DEFAULT_WEIGHTS = (1.0, 0.1, 0.5, 1.0, 0.3, 0.05)


class UniformSampler:
    """Epoch-wise permutations. shard=(r, W): data-parallel rank r of W takes positions r, r+W, ... of the shared
    permutation stream (same rng seed on every rank), so ranks together cover each epoch once."""
    def __init__(self, n, rng, shard=(0, 1)):
        self.n, self.rng = n, rng
        self.r, self.W = shard
        self.order, self.op = [], 0

    def _next(self):
        if self.op >= len(self.order):
            self.order, self.op = self.order[self.op:] + self.rng.permutation(self.n).tolist(), 0
        x = self.order[self.op]
        self.op += 1
        return x

    def sample(self, k):
        out = []
        for _ in range(k):
            for j in range(self.W):
                x = self._next()
                if j == self.r:
                    out.append(x)
        return out

    def update(self, idx, rewards):
        pass

    def stats(self):
        return {}

    def state_dict(self):
        return {"kind": "uniform", "order": list(self.order), "op": self.op, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d):
        assert d["kind"] == "uniform"
        self.order, self.op = list(d["order"]), int(d["op"])
        self.rng.bit_generator.state = d["rng"]


class DifficultyPoolSampler:
    def __init__(self, n, rng, lo=0.25, hi=0.75, weights=DEFAULT_WEIGHTS):
        assert 0 < lo <= hi < 1 and len(weights) == len(POOLS) and min(weights) >= 0
        self.n, self.rng, self.lo, self.hi = n, rng, lo, hi
        self.w = np.asarray(weights, dtype=np.float64)
        self.pool = np.zeros(n, dtype=np.int64)  # index into POOLS; 0 = unseen
        self.last_drawn = np.zeros(len(POOLS), dtype=np.int64)

    def pool_of(self, r):
        if r <= 0:
            return 1
        if r >= 1:
            return 5
        if r <= self.lo:
            return 2
        if r <= self.hi:
            return 3
        return 4

    def probs(self):
        w = self.w[self.pool]
        s = w.sum()
        return w / s if s > 0 else np.full(self.n, 1.0 / self.n)

    def sample(self, k):
        idx = self.rng.choice(self.n, size=k, replace=True, p=self.probs())
        self.last_drawn = np.bincount(self.pool[idx], minlength=len(POOLS))
        return idx.tolist()

    def update(self, idx, rewards):
        """idx: problem indices; rewards: their latest group-mean rewards (a repeated index keeps the last one)."""
        for i, r in zip(idx, rewards):
            self.pool[i] = self.pool_of(float(r))

    def stats(self):
        occ = np.bincount(self.pool, minlength=len(POOLS))
        out = {f"pool_n_{p}": int(c) for p, c in zip(POOLS, occ)}
        tot = max(1, int(self.last_drawn.sum()))
        out.update({f"pool_drawn_{p}": float(c) / tot for p, c in zip(POOLS, self.last_drawn)})
        return out

    def state_dict(self):
        return {"kind": "pool", "pool": self.pool.astype(np.int8), "last_drawn": self.last_drawn.copy(),
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d):
        assert d["kind"] == "pool" and len(d["pool"]) == self.n, "resume: pool size differs"
        self.pool = np.asarray(d["pool"]).astype(np.int64)
        self.last_drawn = np.asarray(d["last_drawn"]).astype(np.int64)
        self.rng.bit_generator.state = d["rng"]


def make_sampler(kind, n, rng, lo=0.25, hi=0.75, weights=DEFAULT_WEIGHTS, shard=(0, 1)):
    if kind == "uniform":
        return UniformSampler(n, rng, shard)
    if kind == "pool":
        return DifficultyPoolSampler(n, rng, lo, hi, weights)
    raise ValueError(kind)


def jumped_state(state, n):
    """A PCG64 bit-generator state moved n jumps ahead (PCG64.jumped): a stream disjoint from the original's start."""
    bg = np.random.PCG64()
    bg.state = state
    return bg.jumped(int(n)).state


def rewidth_sampler_state(saved, fresh, step):
    """Elastic resume (rl_grpo.resume_elastic, world size changed): the state dict to load into a sampler built for
    the NEW rank / world. saved = the old run's rank-0 sampler state; fresh = the new sampler's own state_dict().
    uniform: the rng IS the shared permutation stream (sliced r::W by the new shard), so `saved` as is. pool: the pool
    state (identical on every rank) comes from `saved`; the per-rank draw rng (seed + 1000 r convention) comes from
    `fresh`, jumped by `step` off its step-0 stream."""
    assert saved.get("kind") == fresh.get("kind"), f"resume: sampler kind {saved.get('kind')} vs {fresh.get('kind')}"
    out = dict(saved)
    if saved["kind"] == "pool":
        out["rng"] = jumped_state(fresh["rng"], step)
    return out
