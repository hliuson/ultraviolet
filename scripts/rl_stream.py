"""Streaming rollout generator for rl_grpo.py (--stream).

StreamGen keeps `slots` decode rows busy across optimizer steps. Requests (prompt ids + metadata) wait in a FIFO
queue; every `check_every` decode steps, finished rows (EOS or max_new) are harvested and their slots are immediately
prefilled with queued requests, so there is no per-step tail where most slots idle behind a few long rollouts. The
caller trains whenever enough complete groups are harvested and then calls `sync_weights` (in place: the CUDA graphs
captured on the first chunks keep pointing at the same weight tensors), so in-flight rows continue under the newer
weights. Every generated token's behaviour log-prob (log softmax(logits / T) of the sampled token, under the weights
that produced it) is recorded, so the trainer can apply truncated importance sampling to off-policy tokens.

Mirrors ultraviolet.model.generate_continuous (same harvest / refill / parking logic).
"""
import collections

import numpy as np
import torch
import torch.nn.functional as F

from ultraviolet.model import DecodeRunner, _autocast, _left_pad, sample


@torch.no_grad()
def sync_weights_inplace(gen_m, raw):
    """Copy raw's parameters into gen_m and refresh the fused inference weights *in place* (fuse_for_inference
    allocates new tensors, which would leave captured CUDA graphs reading the old ones)."""
    for pg, pt in zip(gen_m.parameters(), raw.parameters()):
        pg.copy_(pt)
    mods = [m for m in gen_m.modules() if getattr(m, "_fused", None) is not None]
    old = [m._fused for m in mods]
    gen_m.fuse_for_inference(torch.float16)
    for m, o in zip(mods, old):
        new = m._fused
        for k, v in new.items():
            o[k].copy_(v)
        m._fused = o
    torch.cuda.synchronize(gen_m.embed.weight.device)


class StreamGen:
    def __init__(self, model, slots, max_prompt, max_new, temperature, eos_id, check_every=16, compile=True,
                 use_graph=True, pad_id=0, dtype=torch.float16, bucket=1024):
        assert temperature > 0
        self.m, self.B, self.max_new, self.T, self.eos = model, slots, max_new, temperature, eos_id
        self.ce, self.pad, self.dtype = check_every, pad_id, dtype
        self.dev = dev = model.embed.weight.device
        self.max_prompt = max_prompt
        self.st = model.new_state(slots, max_prompt + max_new + check_every, dtype, dev)
        self.runner = DecodeRunner(model, self.st, bucket=bucket, use_graph=use_graph, temperature=temperature,
                                   compile=compile)
        # DecodeRunner.out_logits is the output buffer of the most recently *captured* graph, not of the one just
        # replayed; keep each bucket's buffer (set right after that bucket's capture) to read the right logits
        self.bucket_logits = {}
        W = max_new + check_every
        self.buf = torch.zeros(slots, W, dtype=torch.long, device=dev)
        self.lp = torch.zeros(slots, W, dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(slots, dtype=torch.long, device=dev)
        self.tok = torch.zeros(slots, dtype=torch.long, device=dev)
        self.req = [None] * slots  # (meta, start_version) per row
        self.host_cnt = np.zeros(slots, dtype=np.int64)
        self.queue = collections.deque()
        self.version = 0
        self.n_decode_steps = 0
        self.n_tokens = 0  # generated tokens returned (EOS included)

    # ---------------------------------------------------------------------------------------------- requests
    def add(self, prompt_ids, meta):
        assert len(prompt_ids) <= self.max_prompt, "prompt longer than max_prompt"
        self.queue.append((prompt_ids, meta))

    def n_active(self):
        return sum(r is not None for r in self.req)

    def n_outstanding(self):
        return len(self.queue) + self.n_active()

    def _refill(self, rows):
        rows = rows[:len(self.queue)]
        if not rows:
            return
        items = [self.queue.popleft() for _ in rows]
        idx, starts = _left_pad([p for p, _ in items], self.pad, self.dev)
        rt = torch.tensor(rows, device=self.dev)
        self.buf[rt] = -1
        with _autocast(self.dev, self.dtype):
            logits = self.m.prefill(idx, starts, self.st, rows=rt)
        first = sample(logits, self.T)
        self.lp[rt, 0] = F.log_softmax(logits.float() / self.T, -1).gather(1, first[:, None])[:, 0]
        self.tok[rt] = first
        self.buf[rt, 0] = first
        self.cnt[rt] = 1
        for r, (_, meta) in zip(rows, items):
            self.req[r] = (meta, self.version)
            self.host_cnt[r] = 1

    def _park(self, rows):
        if rows:
            it = torch.tensor(rows, device=self.dev)
            self.st.pos[it] = 0
            self.st.starts[it] = 0
            self.st.host_pos[rows] = 0

    # ---------------------------------------------------------------------------------------------- decode
    def chunk(self):
        """Fill idle rows from the queue, decode check_every steps, harvest finished rows (and refill them).
        Returns [(meta, tokens list (EOS-terminated if it stopped), behaviour logp np array, start_version)]."""
        idle = [r for r in range(self.B) if self.req[r] is None]
        if idle and self.queue:
            self._refill(idle)
        if self.n_active() == 0:
            return []
        B = self.B
        for _ in range(self.ce):
            gl = self.runner.g_len(int(self.st.host_pos.max()))
            tok = self.runner.step(self.tok)
            if self.runner.use_graph:
                lg = self.bucket_logits.setdefault(gl, self.runner.out_logits)
            else:
                lg = self.runner.out_logits
            lpt = F.log_softmax(lg.float() / self.T, -1).gather(1, tok.view(B, 1))
            self.buf.scatter_(1, self.cnt.view(B, 1), tok.view(B, 1))
            self.lp.scatter_(1, self.cnt.view(B, 1), lpt)
            self.cnt.add_(1).clamp_(max=self.buf.size(1) - 1)
            self.tok.copy_(tok)
            self.n_decode_steps += 1
        self.host_cnt += self.ce
        hit = (self.buf == self.eos).any(1).cpu().numpy()
        out, done, rows_buf, rows_lp = [], [], None, None
        for r in range(B):
            if self.req[r] is None:
                continue
            if self.host_cnt[r] >= self.max_new or hit[r]:
                if rows_buf is None:
                    rows_buf, rows_lp = self.buf.cpu().numpy(), self.lp.cpu().numpy()
                n = min(self.host_cnt[r], self.max_new)
                row = rows_buf[r, :n].tolist()
                if self.eos in row:
                    n = row.index(self.eos) + 1
                    row = row[:n]
                meta, v0 = self.req[r]
                out.append((meta, row, rows_lp[r, :n].copy(), v0))
                self.n_tokens += n
                self.req[r] = None
                done.append(r)
        if done:
            self._refill(done)
            self._park([r for r in done if self.req[r] is None])
        return out

    def drop_queued(self, pred):
        """Remove queued (not yet started) requests whose meta satisfies pred; returns how many."""
        keep = [x for x in self.queue if not pred(x[1])]
        n = len(self.queue) - len(keep)
        self.queue = collections.deque(keep)
        return n
