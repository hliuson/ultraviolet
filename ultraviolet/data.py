"""Token shards for math pretraining: encode text -> flat uint16/uint32 .bin shards (EOS after every doc) + meta.json,
and a memmap random-window loader.

  python -m ultraviolet.data encode --tok tok32768/tokenizer.json --inputs 'raw/*.parquet' --out data32k \
      [--val-inputs 'val/*.jsonl' | --val-docs 20000] [--max-bytes 2e9]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

EOS_ID = 0


class ShardWriter:
    def __init__(self, out, split, dtype, shard_tokens):
        self.out, self.split, self.dtype, self.cap = out, split, dtype, shard_tokens
        self.i, self.n_in, self.fh, self.tokens, self.bytes, self.docs = 0, 0, None, 0, 0, 0

    def _open(self):
        self.fh = open(os.path.join(self.out, f"{self.split}_{self.i:04d}.bin"), "wb")
        self.n_in = 0

    def write(self, ids_list, nbytes):
        for ids in ids_list:
            if self.fh is None:
                self._open()
            arr = np.asarray(ids + [EOS_ID], dtype=self.dtype)
            self.fh.write(arr.tobytes())
            self.n_in += len(arr)
            self.tokens += len(arr)
            self.docs += 1
            if self.n_in >= self.cap:
                self.fh.close()
                self.fh = None
                self.i += 1
        self.bytes += nbytes

    def close(self):
        if self.fh is not None:
            self.fh.close()
        return {"tokens": self.tokens, "bytes": self.bytes, "docs": self.docs,
                "bytes_per_token": round(self.bytes / max(1, self.tokens), 4)}


def encode(a):
    from tokenizers import Tokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from train_tokenizer import iter_texts  # noqa: E402

    tok = Tokenizer.from_file(a.tok)
    V = tok.get_vocab_size()
    assert tok.token_to_id("<|endoftext|>") == EOS_ID
    dtype = np.uint16 if V <= 65536 else np.uint32
    os.makedirs(a.out, exist_ok=True)
    writers = {"train": ShardWriter(a.out, "train", dtype, int(a.shard_tokens)),
               "val": ShardWriter(a.out, "val", dtype, int(a.shard_tokens))}

    def run(texts, pick):
        buf = []

        def flush():
            enc = tok.encode_batch(buf)
            by = {"train": ([], 0), "val": ([], 0)}
            for j, (t, e) in enumerate(zip(buf, enc)):
                s = pick()
                ids, nb = by[s]
                ids.append(e.ids)
                by[s] = (ids, nb + len(t.encode("utf-8")))
            for s, (ids, nb) in by.items():
                if ids:
                    writers[s].write(ids, nb)
            buf.clear()

        for t in texts:
            buf.append(t)
            if len(buf) >= 4096:
                flush()
                print({k: w.tokens for k, w in writers.items()}, flush=True)
        if buf:
            flush()

    if a.val_inputs:
        run(iter_texts(a.val_inputs, None, 0), lambda: "val")
        run(iter_texts(a.inputs, a.max_bytes, 0), lambda: "train")
    else:
        cnt = [0]

        def pick():
            cnt[0] += 1
            return "val" if cnt[0] <= a.val_docs else "train"

        run(iter_texts(a.inputs, a.max_bytes, 0), pick)
    meta = {"tokenizer": os.path.abspath(a.tok), "vocab": V, "dtype": np.dtype(dtype).name, "eos_id": EOS_ID,
            "inputs": a.inputs, "val_inputs": a.val_inputs}
    meta.update({k: w.close() for k, w in writers.items()})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


class TraceStream:
    """Epoch-shuffled stream over a small doc set (e.g. teacher traces), for high-repetition mixing.

    `tokens.npy` holds all docs concatenated (each ending in EOS) and `offsets.npy` holds the n+1 doc boundaries.
    Epoch e is the docs in permutation rng([seed, 7919, e]) concatenated, and the infinite stream is epoch 0, 1, ...
    Window j is stream[j*T : j*T + T + 1]. Consecutive windows therefore walk through each epoch once before any
    doc repeats, so repeats are spread evenly through training rather than back-to-back."""

    def __init__(self, data_dir: str, seed: int = 0):
        self.tok = np.load(os.path.join(data_dir, "tokens.npy"))
        self.off = np.load(os.path.join(data_dir, "offsets.npy"))
        self.n = len(self.off) - 1
        self.L = int(self.off[-1])
        self.seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def epoch(self, e: int) -> np.ndarray:
        if e not in self._cache:
            perm = np.random.default_rng([self.seed, 7919, e]).permutation(self.n)
            self._cache[e] = np.concatenate([self.tok[self.off[i]:self.off[i + 1]] for i in perm])
            for k in [k for k in self._cache if k < e - 1]:
                del self._cache[k]
        return self._cache[e]

    def window(self, j: int, T: int) -> np.ndarray:
        o = j * T
        out = []
        need = T + 1
        while need:
            e, w = divmod(o, self.L)
            piece = self.epoch(e)[w:w + need]
            out.append(piece)
            need -= len(piece)
            o += len(piece)
        return np.concatenate(out).astype(np.int64)


def trace_rows(step: int, frac: float, G: int) -> tuple[int, int]:
    """Global trace-window index range [a, b) for `step` when a fraction `frac` of the G sequences per step are
    traces (exact on average: floor(frac*G*step) cumulative)."""
    return int(np.floor(frac * G * step)), int(np.floor(frac * G * (step + 1)))


class TokenShards:
    """Random (T+1)-token windows from memmapped shards of one split (`sample`: any offset; `sample_docs`: windows
    starting at a document / segment start, for document-masked training)."""

    def __init__(self, data_dir: str, split: str):
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        dt = np.dtype(self.meta["dtype"])
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        assert self.files, f"no {split} shards in {data_dir}"
        self.maps = [np.memmap(f, dtype=dt, mode="r") for f in self.files]
        self.lens = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.total = int(self.lens.sum())

    def sample(self, B: int, T: int, rng: np.random.Generator) -> torch.Tensor:
        w = (self.lens - T - 1).clip(min=0).astype(np.float64)
        sh = rng.choice(len(self.maps), size=B, p=w / w.sum())
        out = np.empty((B, T + 1), dtype=np.int64)
        for b, s in enumerate(sh):
            o = int(rng.integers(0, self.lens[s] - T - 1))
            out[b] = self.maps[s][o:o + T + 1]
        return torch.from_numpy(out)

    def segment_starts(self, s: int, max_doc_len: int = 0) -> np.ndarray:
        """Sorted segment-start offsets of shard s (cached): offset 0, every EOS (the EOS between two docs is the
        BOS of the next, as in inference prompts [EOS] + prompt) and, with max_doc_len, every max_doc_len tokens
        inside a longer doc (the chunk boundaries of model.doc_layout)."""
        if not hasattr(self, "_starts"):
            self._starts = {}
        key = (s, max_doc_len)
        if key not in self._starts:
            m = self.maps[s]
            eos = np.flatnonzero(np.asarray(m) == self.meta.get("eos_id", EOS_ID)).astype(np.int64)
            st = np.concatenate([[0], eos[eos > 0]]).astype(np.int64)
            if max_doc_len:
                ln = np.diff(np.concatenate([st, [len(m)]]))
                nc = -(-ln // max_doc_len)
                first = np.repeat(np.cumsum(nc) - nc, nc)
                st = np.repeat(st, nc) + max_doc_len * (np.arange(int(nc.sum())) - first)
            self._starts[key] = st
        return self._starts[key]

    def sample_docs(self, B: int, T: int, rng: np.random.Generator, max_doc_len: int = 0) -> torch.Tensor:
        """Random (T+1)-token windows that start at a segment start (segment_starts): shard by length as `sample`,
        then a uniform segment start among those with a full window. Consecutive docs of the stream fill the row."""
        w = (self.lens - T - 1).clip(min=0).astype(np.float64)
        sh = rng.choice(len(self.maps), size=B, p=w / w.sum())
        out = np.empty((B, T + 1), dtype=np.int64)
        for b, s in enumerate(sh):
            st = self.segment_starts(int(s), max_doc_len)
            n = int(np.searchsorted(st, self.lens[s] - T - 1, side="right"))
            o = int(st[int(rng.integers(0, n))])
            out[b] = self.maps[s][o:o + T + 1]
        return torch.from_numpy(out)

    def fixed_windows(self, n_tokens: int, T: int) -> torch.Tensor:
        """Deterministic windows evenly spaced over the whole split (so every source is represented)."""
        need = max(1, n_tokens // T)
        cap = [max(0, int(n) - T - 1) for n in self.lens]
        tot = sum(cap)
        wins = []
        for i in range(need):
            g = int((i + 0.5) * tot / need)
            for s, c in enumerate(cap):
                if g < c:
                    wins.append(np.asarray(self.maps[s][g:g + T + 1], dtype=np.int64))
                    break
                g -= c
        return torch.from_numpy(np.stack(wins))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("encode")
    e.add_argument("--tok", required=True)
    e.add_argument("--inputs", nargs="+", required=True)
    e.add_argument("--val-inputs", nargs="*", default=None)
    e.add_argument("--val-docs", type=int, default=20000)
    e.add_argument("--out", required=True)
    e.add_argument("--max-bytes", type=float, default=None)
    e.add_argument("--shard-tokens", type=float, default=2 ** 28)
    a = ap.parse_args()
    if a.max_bytes:
        a.max_bytes = int(a.max_bytes)
    encode(a)


if __name__ == "__main__":
    main()
