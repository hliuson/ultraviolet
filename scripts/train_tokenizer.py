"""Train a byte-level BPE tokenizer for the math model (HF tokenizers).

Pretokenization = Qwen2 regex (single-digit numbers, punctuation runs like `}{`, `^{`, `$$` grouped) with LaTeX
commands (` \\frac`, `\\alpha`) kept as whole pretokens + byte-level. Special tokens get the first ids and are
identical across vocab sizes: <|endoftext|>=0, <|im_start|>=1, <|im_end|>=2, <think>=3, </think>=4,
<|pad|>=5, <|reserved_0..63|>=6..69.

Usage:
  python train_tokenizer.py train --inputs 'raw/*.parquet' --out tok32k --vocab 32768 --max-bytes 1e9
  python train_tokenizer.py eval --tok tok32k/tokenizer.json --inputs held.parquet --max-bytes 5e7
Inputs: parquet (column `text`) or jsonl ({"text": ...}); globs ok.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Qwen2 pattern + LaTeX: ` ?\\cmd` is its own pretoken, and punctuation runs never swallow the backslash of a
# following command (so `{\\alpha}` -> `{`, `\\alpha`, `}`).
PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\\\p{L}+|[^\r\n\p{L}\p{N}\\]?\p{L}+|\p{N}"
       r"| ?(?:[^\s\p{L}\p{N}\\]|\\(?!\p{L}))+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|pad|>"] + \
    [f"<|reserved_{i}|>" for i in range(64)]


def iter_texts(paths, max_bytes=None, seed=0, skip_docs=0):
    """Yield doc texts from parquet/jsonl files (file order shuffled by seed) up to max_bytes."""
    files = []
    for p in paths:
        files += sorted(glob.glob(p)) or [p]
    random.Random(seed).shuffle(files)
    nb = 0
    k = 0
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                    k += 1
                    if k <= skip_docs or not t:
                        continue
                    yield t
                    nb += len(t.encode("utf-8"))
                    if max_bytes and nb >= max_bytes:
                        return
        else:
            with open(f) as fh:
                for line in fh:
                    k += 1
                    if k <= skip_docs:
                        continue
                    t = json.loads(line).get("text")
                    if not t:
                        continue
                    yield t
                    nb += len(t.encode("utf-8"))
                    if max_bytes and nb >= max_bytes:
                        return


def iter_balanced(groups, max_bytes, seed=0):
    """Source-balanced sample: each group (comma-separated globs) contributes ~max_bytes/len(groups), with docs
    kept i.i.d. with probability cap/group_bytes so every file of the group is represented."""
    rng = random.Random(seed)
    cap = max_bytes / len(groups)
    for grp in groups:
        files = sorted({f for g in grp.split(",") for f in glob.glob(g)})
        tot = sum(os.path.getsize(f) for f in files)
        p = min(1.0, 1.05 * cap / max(1, tot))
        nb = 0
        rng.shuffle(files)
        for t in iter_texts(files, None, seed):
            if rng.random() < p:
                yield t
                nb += len(t.encode("utf-8"))
                if nb >= cap:
                    break
        print(f"group {grp}: {len(files)} files, {tot / 1e9:.2f} GB on disk, sampled {nb / 1e6:.0f} MB", flush=True)


def build(vocab):
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PAT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2, special_tokens=SPECIALS, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    return tok, tr


def cmd_train(a):
    tok, tr = build(a.vocab)
    t0 = time.time()

    def batches():
        buf = []
        src = iter_balanced(a.groups, a.max_bytes, a.seed) if a.groups else iter_texts(a.inputs, a.max_bytes, a.seed)
        for t in src:
            buf.append(t)
            if len(buf) == 1000:
                yield buf
                buf = []
        if buf:
            yield buf

    tok.train_from_iterator(batches(), trainer=tr)
    os.makedirs(a.out, exist_ok=True)
    tok.save(os.path.join(a.out, "tokenizer.json"))
    try:
        from transformers import PreTrainedTokenizerFast
        hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|pad|>",
                                     additional_special_tokens=SPECIALS[1:5])
        hf.save_pretrained(a.out)
    except Exception as e:  # noqa: BLE001
        print("HF wrapper save failed:", e)
    for i, s in enumerate(SPECIALS[:6]):
        assert tok.token_to_id(s) == i, s
    meta = {"vocab": tok.get_vocab_size(), "train_bytes": a.max_bytes, "inputs": a.inputs, "groups": a.groups,
            "seconds": round(time.time() - t0, 1), "specials": SPECIALS[:6], "n_reserved": 64}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def cmd_eval(a):
    out = {}
    if a.groups:  # per-group bytes/token, then overall on the union
        for grp in a.groups:
            texts = list(iter_balanced([grp], a.max_bytes / len(a.groups), a.seed + 1))
            nb = sum(len(t.encode("utf-8")) for t in texts)
            for p in a.tok:
                tok = Tokenizer.from_file(p)
                n = sum(len(e.ids) for e in tok.encode_batch(texts))
                out.setdefault(grp, {})[p] = round(nb / n, 4)
        a.inputs = [g for grp in a.groups for g in grp.split(",")]
    texts = list(iter_texts(a.inputs, a.max_bytes, a.seed + 1, skip_docs=a.skip_docs))
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    for p in a.tok:
        tok = Tokenizer.from_file(p)
        enc = tok.encode_batch(texts)
        n = sum(len(e.ids) for e in enc)
        rt = all(tok.decode(e.ids, skip_special_tokens=False) == t for e, t in zip(enc[:200], texts[:200]))
        out[p] = {"vocab": tok.get_vocab_size(), "bytes_per_token": round(nbytes / n, 4), "tokens": n,
                  "roundtrip_ok": rt}
    sample = r"Let $x^2 + 12345 = \frac{\alpha}{2}$, so \sqrt{x} \leq 3.14159 and \begin{align} a_{n+1} \end{align}"
    out["example"] = {p: Tokenizer.from_file(p).encode(sample).tokens for p in a.tok}
    out["think_ids"] = {p: Tokenizer.from_file(p).encode("<think>\nok</think>").ids for p in a.tok}
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--inputs", nargs="+", default=None)
    t.add_argument("--groups", nargs="+", default=None, help="source groups (comma-separated globs), balanced")
    t.add_argument("--out", required=True)
    t.add_argument("--vocab", type=int, default=32768)
    t.add_argument("--max-bytes", type=float, default=1e9)
    t.add_argument("--seed", type=int, default=0)
    e = sp.add_parser("eval")
    e.add_argument("--tok", nargs="+", required=True)
    e.add_argument("--inputs", nargs="+", default=None)
    e.add_argument("--groups", nargs="+", default=None)
    e.add_argument("--max-bytes", type=float, default=5e7)
    e.add_argument("--skip-docs", type=int, default=0)
    e.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if getattr(a, "max_bytes", None):
        a.max_bytes = int(a.max_bytes)
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
