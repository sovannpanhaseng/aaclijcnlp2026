#!/usr/bin/env python3
"""
Language Bleed Ratio (LBR), as defined in "Small Model, Native Design".

    LBR = (# generated tokens containing a non-Khmer character)
          / (# generated tokens)

Definition used here (edit the constants below if your paper differs):
  * Khmer characters: U+1780-U+17FF (Khmer) and U+19E0-U+19FF (Khmer Symbols).
    Khmer punctuation (e.g. U+17D4) and Khmer digits (U+17E0-U+17E9) sit
    inside the Khmer block, so they count as Khmer.  ASCII digits, ASCII
    punctuation, Latin, Thai, CJK, Tamil, etc. count as non-Khmer.
  * Whitespace is ignored.  Zero-width characters (U+200B/C/D, U+FEFF) are
    also ignored, because Khmer text commonly uses U+200B as a word separator.
  * A token is counted in the denominator only if it has at least one
    non-ignorable character.  It is flagged as bleed if ANY such character
    is non-Khmer.
  * Byte-fallback tokens (<0xNN>, or tokens that decode to U+FFFD alone) are
    grouped with their neighbours and decoded together, so the three bytes of
    one Khmer character are judged as Khmer, not as three bleed tokens.
  * Thinking traces / channel tags are stripped before scoring (--no-strip
    to disable).

Usage
  # token-level LBR with Bayon's tokenizer, several passes at once
  python lbr.py gens_pass1.jsonl gens_pass2.jsonl gens_pass3.jsonl \
      --tokenizer attentionlab/bayon-it --field output --json out.json

  # baseline with its own tokenizer
  python lbr.py gpt_oss_gens.jsonl --tokenizer openai/gpt-oss-120b

  # no tokenizer: character-level proxy
  python lbr.py gens.jsonl --char-level

  # audit the vocabulary (answers "does it contain Latin/Thai/digit tokens?")
  python lbr.py --audit-vocab --tokenizer attentionlab/bayon-it

Input: .jsonl, .json (list, or dict containing a list) or .csv.  The text
field is auto-detected from: output, generation, response, answer,
prediction, completion, text.  Use --field to force one.
"""

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter

KHMER_RANGES = [(0x1780, 0x17FF), (0x19E0, 0x19FF)]
IGNORABLE_CP = {0x200B, 0x200C, 0x200D, 0xFEFF}
BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
FIELD_CANDIDATES = [
    "output",
    "generation",
    "response",
    "answer",
    "prediction",
    "completion",
    "text",
]


# ---------------------------------------------------------------- characters
def is_khmer(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in KHMER_RANGES)


def is_ignorable(ch):
    return ch.isspace() or ord(ch) in IGNORABLE_CP


def significant(text):
    return [c for c in text if not is_ignorable(c)]


def is_countable(text):
    return len(significant(text)) > 0


def is_bleed(text):
    return any(not is_khmer(c) for c in significant(text))


def script_of(ch):
    cp = ord(ch)
    if is_khmer(ch):
        return "khmer"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isascii() and ch.isdigit():
        return "digit"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    if cat.startswith("L"):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        if name.startswith("LATIN"):
            return "latin"
        if name.startswith("CJK"):
            return "cjk"
        if name.startswith("TAMIL"):
            return "tamil"
    return "other"


# ---------------------------------------------------------------- stripping
THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.S),
    re.compile(r"<thinking>.*?</thinking>", re.S),
    re.compile(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>", re.S),
]
FINAL_MARKER = "<|channel|>final<|message|>"
TAG_RE = re.compile(r"<\|[^|>]*\|>")


def strip_reasoning(text):
    if FINAL_MARKER in text:
        text = text.split(FINAL_MARKER)[-1]
    for p in THINK_PATTERNS:
        text = p.sub("", text)
    if "</think>" in text:  # unbalanced: keep what follows the last close
        text = text.split("</think>")[-1]
    text = TAG_RE.sub("", text)
    text = re.sub(r"</?(?:bos|eos|pad)>", "", text)
    return text.strip()


# ---------------------------------------------------------------- tokenizing
def load_tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def tokenize_units(tok, text):
    """Return a list of units: (n_tokens, decoded_text, n_byte_fallback)."""
    special = set(getattr(tok, "all_special_ids", []) or [])
    ids = [i for i in tok.encode(text, add_special_tokens=False) if i not in special]
    info = []
    for i in ids:
        raw = tok.convert_ids_to_tokens(i)
        dec = tok.decode(
            [i], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        is_byte = bool(BYTE_RE.match(raw or "")) or "\ufffd" in dec
        info.append((i, dec, is_byte))

    units, k = [], 0
    while k < len(info):
        i, dec, is_byte = info[k]
        if not is_byte:
            units.append((1, dec, 0))
            k += 1
            continue
        j = k
        while j < len(info) and info[j][2]:
            j += 1
        group = [x[0] for x in info[k:j]]
        merged = tok.decode(
            group, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        units.append((len(group), merged, len(group)))
        k = j
    return units


def char_units(text):
    return [(1, c, 0) for c in text if not is_ignorable(c)]


# ---------------------------------------------------------------- scoring
def score_text(text, tok, char_level, strip):
    if strip:
        text = strip_reasoning(text)
    units = char_units(text) if char_level else tokenize_units(tok, text)
    total = bleed = byte_fb = 0
    flagged = Counter()
    scripts = Counter()
    for n, s, nb in units:
        if not is_countable(s):
            continue
        total += n
        byte_fb += nb
        if is_bleed(s):
            bleed += n
            flagged[s] += n
            for c in significant(s):
                if not is_khmer(c):
                    scripts[script_of(c)] += 1
    return total, bleed, byte_fb, flagged, scripts


# ---------------------------------------------------------------- input
def pick_field(rec, field):
    if field:
        return rec[field]
    for f in FIELD_CANDIDATES:
        if f in rec and isinstance(rec[f], str):
            return rec[f]
    raise KeyError(f"no text field found; keys are {list(rec)}; use --field")


def parse_bench_txt(path):
    """Parse the '--- Q&A Pair #N ---' / 'Question:' / 'Response:' format
    written by bench.py."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    parts = re.split(r"--- Q&A Pair #(\d+) ---\n", raw)
    recs = []
    for k in range(1, len(parts), 2):
        body = parts[k + 1].split("=" * 60)[0]
        m = re.search(r"Question:(.*?)\nResponse:(.*)", body, re.S)
        if m:
            recs.append(
                {
                    "index": int(parts[k]),
                    "question": m.group(1).strip(),
                    "output": m.group(2).strip(),
                }
            )
        else:
            print(
                f"warning: could not parse pair #{parts[k]} in {path}", file=sys.stderr
            )
    print(f"{path}: parsed {len(recs)} pairs", file=sys.stderr)
    return recs


def load_records(path):
    p = path.lower()
    if p.endswith(".txt"):
        return parse_bench_txt(path)
    if p.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if p.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            raise ValueError(f"{path}: no list found in JSON object")
        return data
    if p.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported file type: {path}")


# ---------------------------------------------------------------- vocab audit
def audit_vocab(tok, out):
    counts = Counter()
    examples = {}
    special = set(getattr(tok, "all_special_tokens", []) or [])
    for raw, i in tok.get_vocab().items():
        if raw in special:
            counts["special"] += 1
            continue
        if BYTE_RE.match(raw):
            counts["byte_fallback"] += 1
            continue
        try:
            s = tok.convert_tokens_to_string([raw])
        except Exception:
            s = raw
        sig = significant(s)
        if not sig:
            counts["whitespace_only"] += 1
            continue
        kinds = {script_of(c) for c in sig}
        key = (
            "khmer_only"
            if kinds == {"khmer"}
            else "non_khmer:" + "+".join(sorted(kinds - {"khmer"}))
        )
        counts[key] += 1
        if key != "khmer_only":
            examples.setdefault(key, []).append(s)
    print(f"Vocabulary size: {len(tok.get_vocab())}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v}")
        if k in examples:
            print("      e.g. " + ", ".join(repr(x) for x in examples[k][:8]))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"counts": counts, "examples": examples},
                f,
                ensure_ascii=False,
                indent=2,
            )


# ---------------------------------------------------------------- main
def bootstrap_ci(per_gen, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(per_gen)
    vals = []
    for _ in range(n_boot):
        b = t = 0
        for _ in range(n):
            x = per_gen[rng.randrange(n)]
            b += x[0]
            t += x[1]
        vals.append(100.0 * b / t if t else 0.0)
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--field")
    ap.add_argument("--tokenizer")
    ap.add_argument("--char-level", action="store_true")
    ap.add_argument("--no-strip", action="store_true")
    ap.add_argument("--audit-vocab", action="store_true")
    ap.add_argument("--json", help="write full results to this file")
    ap.add_argument("--flagged-csv", help="write every flagged token + context")
    a = ap.parse_args()

    if a.audit_vocab:
        if not a.tokenizer:
            sys.exit("--audit-vocab needs --tokenizer")
        audit_vocab(load_tokenizer(a.tokenizer), a.json)
        return

    if not a.files:
        sys.exit("give at least one generations file")
    if not a.char_level and not a.tokenizer:
        sys.exit("give --tokenizer (token-level LBR) or --char-level")
    tok = None if a.char_level else load_tokenizer(a.tokenizer)

    T = B = BF = 0
    flagged_all, scripts_all = Counter(), Counter()
    per_gen, rows, n_gen, n_gen_bleed = [], [], 0, 0
    for path in a.files:
        for idx, rec in enumerate(load_records(path)):
            text = pick_field(rec, a.field)
            t, b, bf, fl, sc = score_text(text, tok, a.char_level, not a.no_strip)
            T += t
            B += b
            BF += bf
            flagged_all.update(fl)
            scripts_all.update(sc)
            per_gen.append((b, t))
            n_gen += 1
            n_gen_bleed += b > 0
            for s, c in fl.items():
                rows.append({"file": path, "index": idx, "token": s, "count": c})

    lbr = 100.0 * B / T if T else 0.0
    lo, hi = bootstrap_ci(per_gen) if per_gen and T else (0.0, 0.0)
    unit = "chars" if a.char_level else "tokens"
    print(f"Generations scored : {n_gen}  (files: {len(a.files)})")
    print(f"Total {unit:<12s}: {T:,}")
    print(f"Bleed {unit:<12s}: {B:,}")
    print(
        f"LBR                : {lbr:.4f}%   (95% bootstrap CI over generations: {lo:.4f}-{hi:.4f}%)"
    )
    print(f"Byte-fallback      : {BF:,} ({100.0 * BF / T if T else 0:.2f}%)")
    print(f"Generations with any bleed: {n_gen_bleed}/{n_gen}")
    print("Bleed by script    :", dict(scripts_all))
    print("Top flagged tokens :")
    for s, c in flagged_all.most_common(25):
        print(f"   {c:6d}  {s!r}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generations": n_gen,
                    "total": T,
                    "bleed": B,
                    "lbr_percent": lbr,
                    "ci95": [lo, hi],
                    "byte_fallback": BF,
                    "scripts": scripts_all,
                    "flagged": flagged_all,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    if a.flagged_csv:
        with open(a.flagged_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "index", "token", "count"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
