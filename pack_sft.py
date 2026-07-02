import json
import os
import random

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
INPUT_JSONL_FILES = [
    "./sft.jsonl",
]
TOKENIZER_MODEL = "tokenizerv1.model"

OUTPUT_TRAIN_BIN_FILE = "./data/sft_packed_train.bin"
OUTPUT_EVAL_BIN_FILE = "./data/sft_packed_eval.bin"
# NEW: Mask output files
OUTPUT_TRAIN_MASK_FILE = "./data/sft_packed_train.mask.bin"
OUTPUT_EVAL_MASK_FILE = "./data/sft_packed_eval.mask.bin"

SEQ_LEN = 1024
BOS_ID = 1
EOS_ID = 2
EVAL_RATIO = 0.05

random.seed(61)

# ==========================================
# 2. Initialization
# ==========================================
print("[System] Loading customized Tokenizer...")
if not os.path.exists(TOKENIZER_MODEL):
    raise FileNotFoundError(
        f"[Error] Could not find tokenizer model: {TOKENIZER_MODEL}"
    )

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

train_buffer = []
eval_buffer = []

# NEW: Buffers to hold our 1s and 0s
train_mask_buffer = []
eval_mask_buffer = []

train_documents = 0
eval_documents = 0

saved_train_blocks = 0
saved_eval_blocks = 0

# ==========================================
# 3. Processing and Packing Loop
# ==========================================
print("[System] Starting token-efficient SFT packing with answer-only masking...")

# Ensure output directory exists
os.makedirs("./data", exist_ok=True)

with (
    open(OUTPUT_TRAIN_BIN_FILE, "wb") as f_train,
    open(OUTPUT_EVAL_BIN_FILE, "wb") as f_eval,
    open(OUTPUT_TRAIN_MASK_FILE, "wb") as f_train_mask,  # Open mask files
    open(OUTPUT_EVAL_MASK_FILE, "wb") as f_eval_mask,
):
    for file_path in INPUT_JSONL_FILES:
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}. Skipping.")
            continue

        print(f"\n[System] Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Tokenizing {os.path.basename(file_path)}"):
                try:
                    data = json.loads(line)

                    question = data.get("question", "").strip()
                    answer = data.get("answer", "").strip()

                    if not question or not answer:
                        continue

                    # Step 1: Tokenize question and answer separately to know the boundary
                    q_tokens = sp.encode_as_ids(f"{question}\n")
                    a_tokens = sp.encode_as_ids(answer)

                    # Step 2: Combine tokens and create the parallel mask
                    final_tokens = [BOS_ID] + q_tokens + a_tokens + [EOS_ID]

                    # Mask logic:
                    # BOS (0) + Question (0s) + Answer (1s) + EOS (1)
                    # We want the model to predict the answer and when to output EOS
                    final_mask = [0] + [0] * len(q_tokens) + [1] * len(a_tokens) + [1]

                    # Step 3: Route to either Eval or Train
                    is_eval = random.random() < EVAL_RATIO

                    if is_eval:
                        eval_buffer.extend(final_tokens)
                        eval_mask_buffer.extend(final_mask)  # Add to mask buffer
                        eval_documents += 1

                        while len(eval_buffer) >= SEQ_LEN:
                            block = eval_buffer[:SEQ_LEN]
                            mask_block = eval_mask_buffer[:SEQ_LEN]

                            eval_buffer = eval_buffer[SEQ_LEN:]
                            eval_mask_buffer = eval_mask_buffer[SEQ_LEN:]

                            # Write tokens (uint16) and masks (uint8 is enough for 0/1)
                            f_eval.write(np.array(block, dtype=np.uint16).tobytes())
                            f_eval_mask.write(
                                np.array(mask_block, dtype=np.uint8).tobytes()
                            )
                            saved_eval_blocks += 1
                    else:
                        train_buffer.extend(final_tokens)
                        train_mask_buffer.extend(final_mask)  # Add to mask buffer
                        train_documents += 1

                        while len(train_buffer) >= SEQ_LEN:
                            block = train_buffer[:SEQ_LEN]
                            mask_block = train_mask_buffer[:SEQ_LEN]

                            train_buffer = train_buffer[SEQ_LEN:]
                            train_mask_buffer = train_mask_buffer[SEQ_LEN:]

                            f_train.write(np.array(block, dtype=np.uint16).tobytes())
                            f_train_mask.write(
                                np.array(mask_block, dtype=np.uint8).tobytes()
                            )
                            saved_train_blocks += 1

                except json.JSONDecodeError:
                    continue

# ==========================================
# 4. Summary
# ==========================================
print("\n[Success] SFT Packing with Answer-Only Masks Complete!")
print("-" * 50)
print(
    f"-> Train Tokens: {saved_train_blocks * SEQ_LEN:,} (Masks saved to {OUTPUT_TRAIN_MASK_FILE})"
)
print(
    f"-> Eval Tokens: {saved_eval_blocks * SEQ_LEN:,} (Masks saved to {OUTPUT_EVAL_MASK_FILE})"
)
print("-" * 50)
