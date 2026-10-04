#!/usr/bin/env python3
"""
Corpus-level tokenizer comparison for the Bayon paper.

For every tokenizer it reports, over the same text:
  * total tokens
  * byte-fragment tokens (byte-fallback <0xNN> tokens, or byte-level-BPE tokens
    whose bytes are not valid UTF-8 on their own, e.g. 1-2 bytes of a 3-byte
    Khmer character)
  * <unk> tokens
  * tokens per 1,000 characters, characters per token, UTF-8 bytes per token
  * ratio of tokens vs. the first tokenizer (Bayon), with a 95% bootstrap CI
    over batches of documents
It also reports the character composition of the corpus (Khmer / Latin /
digits / Thai / ASCII punctuation / other), which is useful context for LBR.

Usage
  pip install numpy transformers tiktoken pyarrow
  python corpus_tokenizer_stats.py --input corpus_dir_or_file [more ...] \
      --text-field text --bayon attentionlab/bayon \
      --tiktoken o200k_base --hf Qwen/Qwen2.5-0.5B \
      --out corpus_stats.json

  # quick run on every 20th document first
  python corpus_tokenizer_stats.py --input corpus/ --stride 20

Inputs: .parquet / .jsonl / .json (field --text-field) or .txt (one document
per non-empty line); directories are searched recursively.

Caveat for the paper: if Bayon's tokenizer was trained on this corpus, the
result is in-sample for Bayon.  Also run a held-out text (e.g. a FineWeb-2
Khmer test split, or Khmer Wikipedia) with --input and report both.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
KH_RE = re.compile("[\u1780-\u17ff\u19e0-\u19ff]")
WS_RE = re.compile("[\\s\u200b\u200c\u200d\ufeff]")
LATIN_RE = re.compile("[A-Za-z]")
DIGIT_RE = re.compile("[0-9]")
THAI_RE = re.compile("[\u0e00-\u0e7f]")
PUNCT_RE = re.compile("[!-/:-@\\[-`{-~]")


# ---------------------------------------------------------------- input
def iter_docs(paths, field):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("parquet", "jsonl", "json", "txt"):
                files += sorted(
                    glob.glob(os.path.join(p, "**", "*." + ext), recursive=True)
                )
        else:
            files.append(p)
    if not files:
        sys.exit("no input files found")
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1000, columns=[field]):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
        elif f.endswith(".jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        t = rec.get(field) if isinstance(rec, dict) else rec
                        if t:
                            yield t
        elif f.endswith(".json"):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            for rec in data:
                t = rec.get(field) if isinstance(rec, dict) else rec
                if t:
                    yield t
        else:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield line.rstrip("\n")


# ---------------------------------------------------------------- tokenizers
def _byte_unicode_inverse():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def fragment_table_hf(tok):
    vocab = tok.get_vocab()
    frag = np.zeros(max(vocab.values()) + 1, dtype=bool)
    bl = "ByteLevel" in (
        str(getattr(getattr(tok, "backend_tokenizer", None), "pre_tokenizer", ""))
        + str(getattr(getattr(tok, "backend_tokenizer", None), "decoder", ""))
    )
    inv = _byte_unicode_inverse() if bl else {}
    for s, i in vocab.items():
        if BYTE_RE.match(s):
            frag[i] = True
        elif bl and all(c in inv for c in s):
            try:
                bytes(inv[c] for c in s).decode("utf-8")
            except UnicodeDecodeError:
                frag[i] = True
    return frag


class HFCounter:
    def __init__(self, name):
        from transformers import AutoTokenizer

        self.label = "hf:" + name
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.tok.model_max_length = int(1e9)
        self.frag = fragment_table_hf(self.tok)
        self.unk = self.tok.unk_token_id

    def count(self, texts):
        n = f = u = 0
        for ids in self.tok(texts, add_special_tokens=False)["input_ids"]:
            if not ids:
                continue
            a = np.asarray(ids)
            n += len(a)
            f += int(self.frag[a].sum())
            if self.unk is not None:
                u += int((a == self.unk).sum())
        return n, f, u


class TikCounter:
    def __init__(self, enc, label, threads):
        self.label = label
        self.enc = enc
        self.threads = threads
        self.frag = np.zeros(enc.n_vocab, dtype=bool)
        for i in range(enc.n_vocab):
            try:
                b = enc.decode_single_token_bytes(i)
            except Exception:
                continue
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                self.frag[i] = True

    def count(self, texts):
        n = f = 0
        for ids in self.enc.encode_ordinary_batch(texts, num_threads=self.threads):
            if not ids:
                continue
            a = np.asarray(ids)
            n += len(a)
            f += int(self.frag[a].sum())
        return n, f, 0


def make_tiktoken(name, threads):
    import tiktoken

    return TikCounter(tiktoken.get_encoding(name), "tiktoken:" + name, threads)


# ---------------------------------------------------------------- composition
def _n(rx, s):
    return len(s) - len(rx.sub("", s))


def composition(t):
    rest = KH_RE.sub("", t)
    khmer = len(t) - len(rest)
    rest2 = WS_RE.sub("", rest)
    latin = _n(LATIN_RE, rest2)
    digit = _n(DIGIT_RE, rest2)
    thai = _n(THAI_RE, rest2)
    punct = _n(PUNCT_RE, rest2)
    other = len(rest2) - latin - digit - thai - punct
    return {
        "khmer": khmer,
        "latin": latin,
        "ascii_digit": digit,
        "thai": thai,
        "ascii_punct": punct,
        "other": other,
        "zwsp": t.count("\u200b"),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--bayon", default="attentionlab/bayon")
    ap.add_argument("--hf", action="append", default=[])
    ap.add_argument("--tiktoken", action="append", default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every K-th document")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--batch-docs", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()

    counters = [HFCounter(a.bayon)] + [HFCounter(h) for h in a.hf]
    for name in a.tiktoken or ["o200k_base"]:
        counters.append(make_tiktoken(name, a.threads))
    labels = [c.label for c in counters]

    tot = {l: Counter() for l in labels}
    comp = Counter()
    rows = []  # per batch: [chars, bytes, then (tokens) per counter]
    chars = nbytes = docs = 0
    t0 = time.time()

    def flush(batch):
        nonlocal chars, nbytes, docs
        c = sum(len(t) for t in batch)
        b = sum(len(t.encode("utf-8")) for t in batch)
        chars += c
        nbytes += b
        docs += len(batch)
        for t in batch:
            comp.update(composition(t))
        row = [c, b]
        for cn in counters:
            n, f, u = cn.count(batch)
            tot[cn.label].update({"tokens": n, "fragments": f, "unk": u})
            row.append(n)
        rows.append(row)
        if len(rows) % 20 == 0:
            print(
                f"[{time.time() - t0:7.0f}s] {docs:,} docs, {chars:,} chars",
                file=sys.stderr,
            )

    batch = []
    used = 0
    for k, doc in enumerate(iter_docs(a.input, a.text_field)):
        if k % a.stride:
            continue
        if a.max_docs and used >= a.max_docs:
            break
        batch.append(doc)
        used += 1
        if len(batch) >= a.batch_docs:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    if not rows:
        sys.exit("no documents read")

    arr = np.asarray(rows, dtype=np.float64)  # (B, 2 + n_counters)
    rng = np.random.default_rng(0)
    B = len(arr)
    boots = []
    for _ in range(1000):
        boots.append(arr[rng.integers(0, B, B)].sum(0))
    boots = np.asarray(boots)

    print(f"\nDocuments: {docs:,}   characters: {chars:,}   UTF-8 bytes: {nbytes:,}")
    print(
        f"{'tokenizer':32s} {'tokens':>15s} {'frag%':>7s} {'unk':>6s} "
        f"{'tok/1k ch':>10s} {'ch/tok':>7s} {'B/tok':>6s} {'ratio vs Bayon (95% CI)':>28s}"
    )
    result = {"documents": docs, "characters": chars, "bytes": nbytes, "tokenizers": {}}
    for j, l in enumerate(labels):
        n = tot[l]["tokens"]
        f = tot[l]["fragments"]
        u = tot[l]["unk"]
        ratio = boots[:, 2 + j] / boots[:, 2]
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        r = n / tot[labels[0]]["tokens"]
        print(
            f"{l:32s} {n:15,d} {100 * f / n:7.2f} {u:6d} {1000 * n / chars:10.1f} "
            f"{chars / n:7.2f} {nbytes / n:6.2f} {r:10.2f}x ({lo:.2f}-{hi:.2f})"
        )
        result["tokenizers"][l] = {
            "tokens": n,
            "fragment_tokens": f,
            "unk": u,
            "tokens_per_1k_chars": 1000 * n / chars,
            "ratio_vs_first": r,
            "ratio_ci95": [lo, hi],
        }

    nonws = sum(v for k, v in comp.items() if k != "zwsp")
    print("\nRaw counts")
    print(f"  documents                      {docs:>15,d}")
    print(f"  characters (all)               {chars:>15,d}")
    print(f"  characters (no whitespace/ZW)  {nonws:>15,d}")
    print(f"  UTF-8 bytes                    {nbytes:>15,d}")
    print("\nCharacter composition (excluding whitespace and zero-width):")
    for k in ("khmer", "latin", "ascii_digit", "thai", "ascii_punct", "other"):
        print(f"  {k:12s} {comp[k]:>15,d}  {100 * comp[k] / nonws:7.3f}%")
    print(f"  U+200B (zero-width space) count: {comp['zwsp']:,}")
    result["composition"] = dict(comp)
    result["non_whitespace_characters"] = nonws
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
