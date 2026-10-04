import json
import os
import random

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
INPUT_JSONL_FILE = "./input.jsonl"
TOKENIZER_MODEL = "../tokenizerv1_extended.model"
OUTPUT_TRAIN_BIN_FILE = "../data/sft_packed_train.bin"
OUTPUT_EVAL_BIN_FILE = "../data/sft_packed_eval.bin"
OUTPUT_TRAIN_MASK_FILE = "../data/sft_packed_train.mask.bin"
OUTPUT_EVAL_MASK_FILE = "../data/sft_packed_eval.mask.bin"
SEQ_LEN = 1024
BOS_ID = 1
PAD_ID = 0
EVAL_RATIO = 0.05
random.seed(61)


# ==========================================
# Helper Functions
# ==========================================
def tokenize_khmer_turn(question, answer):
    """Tokenizes a single Q&A pair. EOS assumed already present in source text."""
    q_tokens = sp.encode_as_ids(question)
    sep_tokens = sp.encode_as_ids("")
    a_tokens = sp.encode_as_ids(answer)
    tokens = [BOS_ID] + q_tokens + sep_tokens + a_tokens
    masks = [0] + [0] * len(q_tokens) + [0] * len(sep_tokens) + [1] * len(a_tokens)
    return tokens, masks


def write_document(tokens, masks, output_file, mask_file):
    """Writes a single document (Q&A pair) to binary files with padding."""
    seq_len = len(tokens)
    # Pad with PAD_ID, not EOS -- EOS now appears exactly once, embedded in
    # the source answer text, and padding shouldn't repeat it.
    padded_tokens = tokens + [PAD_ID] * (SEQ_LEN - seq_len)
    padded_masks = masks + [0] * (SEQ_LEN - seq_len)
    output_file.write(np.array(padded_tokens, dtype=np.uint16).tobytes())
    mask_file.write(np.array(padded_masks, dtype=np.uint8).tobytes())


# ==========================================
# Initialization
# ==========================================
print("[System] Loading tokenizer...")
if not os.path.exists(TOKENIZER_MODEL):
    raise FileNotFoundError(
        f"[Error] Could not find tokenizer model: {TOKENIZER_MODEL}"
    )
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

train_documents = 0
eval_documents = 0
skipped_documents = 0

# ==========================================
# Processing
# ==========================================
print("[System] Starting SFT dataset processing...")
os.makedirs("../data", exist_ok=True)

if not os.path.exists(INPUT_JSONL_FILE):
    raise FileNotFoundError(f"[Error] Input file not found: {INPUT_JSONL_FILE}")

print(f"\n[System] Processing: {INPUT_JSONL_FILE}")
print("[Note] Each Q&A pair is treated as one individual document")
print(f"[Note] Documents longer than {SEQ_LEN} tokens will be skipped")

with (
    open(OUTPUT_TRAIN_BIN_FILE, "wb") as f_train,
    open(OUTPUT_EVAL_BIN_FILE, "wb") as f_eval,
    open(OUTPUT_TRAIN_MASK_FILE, "wb") as f_train_mask,
    open(OUTPUT_EVAL_MASK_FILE, "wb") as f_eval_mask,
    open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_jsonl,
):
    for line in tqdm(f_jsonl, desc="Processing documents"):
        try:
            data = json.loads(line)
            question = data.get("question", "").strip()
            answer = data.get("answer", "").strip()

            if not question or not answer:
                continue

            tokens, masks = tokenize_khmer_turn(question, answer)

            # Skip documents that exceed context limit
            if len(tokens) > SEQ_LEN:
                skipped_documents += 1
                continue

            is_eval = random.random() < EVAL_RATIO

            if is_eval:
                write_document(tokens, masks, f_eval, f_eval_mask)
                eval_documents += 1
            else:
                write_document(tokens, masks, f_train, f_train_mask)
                train_documents += 1

        except json.JSONDecodeError:
            continue


# ==========================================
# Summary
# ==========================================
print("\n[Success] Processing complete!")
print("-" * 60)
print(f"Train documents: {train_documents:,}")
print(f"Eval documents: {eval_documents:,}")
print(f"Skipped (too long): {skipped_documents:,}")
print(f"Total valid documents: {train_documents + eval_documents:,}")
print("-" * 60)
