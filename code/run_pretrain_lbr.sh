#!/usr/bin/env bash
# LBR of the raw pre-trained checkpoint (no SFT) on the 201 benchmark questions.
# Needs bench_base.py, lbr.py and inference.py in the same folder.
#   ./run_pretrain_lbr.sh checkpoints/<pretrain_step18k>.pt            # 1,024 new tokens
#   ./run_pretrain_lbr.sh checkpoints/<pretrain_step18k>.pt 256        # faster
set -e
CKPT="${1:?usage: ./run_pretrain_lbr.sh path/to/pretrain.pt [max_new_tokens]}"
MAXTOK="${2:-1024}"
TOK=attentionlab/bayon
for s in 0 1 2; do
  python bench_base.py --checkpoint "$CKPT" --seed "$s" \
      --max-new-tokens "$MAXTOK" --out "pretrain_pass$((s+1)).txt"
done
python lbr.py pretrain_pass1.txt pretrain_pass2.txt pretrain_pass3.txt \
    --tokenizer "$TOK" --json pretrain_lbr.json --flagged-csv pretrain_flagged.csv
python lbr.py pretrain_pass1.txt pretrain_pass2.txt pretrain_pass3.txt --char-level
