#!/usr/bin/env bash
# Pass the pre-training corpus (.jsonl) through the tokenizers and compare:
# tokens, byte-fragment share, tokens per 1,000 characters, raw character counts.
# Needs corpus_tokenizer_stats.py in the same folder.
#
#   ./run_corpus_stats.sh path/to/corpus_dir            # full corpus, field "text"
#   ./run_corpus_stats.sh path/to/corpus_dir text 20    # every 20th document (test first)
#   ./run_corpus_stats.sh a.jsonl,b.jsonl text 1        # several files/folders, comma-separated
#                                                       # (counted together as ONE corpus; run
#                                                       #  separately for main vs held-out)
#
# Optional environment variables:
#   BAYON_TOK=attentionlab/bayon   QWEN_TOK=Qwen/Qwen2.5-0.5B   THREADS=8
#   EXTRA_HF=google/gemma-2-2b     (gated: accept the license on Hugging Face, then `huggingface-cli login`)
# Put only corpus files in the input folder (.jsonl/.json/.parquet/.txt are all read).
set -euo pipefail

INPUT="${1:?usage: ./run_corpus_stats.sh <jsonl_dir_or_file> [text_field] [stride]}"
FIELD="${2:-text}"
STRIDE="${3:-1}"
BAYON_TOK="${BAYON_TOK:-attentionlab/bayon}"
QWEN_TOK="${QWEN_TOK:-Qwen/Qwen2.5-0.5B}"
EXTRA_HF="${EXTRA_HF:-}"
THREADS="${THREADS:-8}"
OUT="corpus_stats_stride${STRIDE}.json"

python3 -c "import numpy, transformers, tiktoken" 2>/dev/null \
  || pip install numpy transformers tiktoken pyarrow

python3 - "$INPUT" "$FIELD" <<'PY'
import glob, json, os, sys
paths, field = sys.argv[1].split(","), sys.argv[2]
total = 0
for p in paths:
    files = [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
    if not files:
        sys.exit(f"no .jsonl files under {p}")
    with open(files[0], encoding="utf-8") as f:
        rec = json.loads(f.readline())
    if field not in rec:
        sys.exit(f"field '{field}' not in {files[0]}; keys are {list(rec)}. Pass the right one as the 2nd argument.")
    print(f"ok: {p}: {len(files)} jsonl file(s)")
    total += len(files)
print(f"{total} file(s) in total, field '{field}'")
PY
IFS=',' read -r -a INPUTS <<< "$INPUT"
ARGS=(--input "${INPUTS[@]}" --text-field "$FIELD" --bayon "$BAYON_TOK"
      --hf "$QWEN_TOK" --tiktoken o200k_base
      --stride "$STRIDE" --threads "$THREADS" --out "$OUT")
if [ -n "$EXTRA_HF" ]; then ARGS+=(--hf "$EXTRA_HF"); fi

python3 corpus_tokenizer_stats.py "${ARGS[@]}" 2>&1 | tee "${OUT%.json}.log"
echo "saved: $OUT and ${OUT%.json}.log"
