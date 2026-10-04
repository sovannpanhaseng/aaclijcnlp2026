import json
import os
import random

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
# List all your JSONL files here to combine them into one dataset
INPUT_JSONL_FILES = [
    "./data/khmer_corpus_normal.jsonl",
    "./data/khmer_corpus_recovered.jsonl",
]
TOKENIZER_MODEL = "gemma_khmer_5k.model"

# Output files for train and eval
OUTPUT_TRAIN_BIN_FILE = "./data/pretrain_packed_data.bin"
OUTPUT_EVAL_BIN_FILE = "./data/eval_packed_data.bin"

SEQ_LEN = 1024  # Match your model's maximum context window length
EOS_ID = 2  # The exact EOS ID specified in your tokenizer script
EVAL_RATIO = 0.05  # 5% of the data will go to the evaluation set

# Set a random seed for reproducible train/eval splits
random.seed(31)

# ==========================================
# 2. Initialization
# ==========================================
print("[System] Loading customized Khmer Tokenizer...")
if not os.path.exists(TOKENIZER_MODEL):
    raise FileNotFoundError(
        f"[Error] Could not find tokenizer model: {TOKENIZER_MODEL}"
    )

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

# Separate buffers and counters for Train and Eval
train_buffer = []
eval_buffer = []

train_documents = 0
eval_documents = 0

saved_train_blocks = 0
saved_eval_blocks = 0

# ==========================================
# 3. Processing and Packing Loop
# ==========================================
print("[System] Starting tokenization and splitting into Train/Eval sets...")

# Open both binary files in write-binary mode
with (
    open(OUTPUT_TRAIN_BIN_FILE, "wb") as f_train,
    open(OUTPUT_EVAL_BIN_FILE, "wb") as f_eval,
):
    # Iterate through all provided JSONL files
    for file_path in INPUT_JSONL_FILES:
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}. Skipping.")
            continue

        print(f"\n[System] Processing {file_path}...")

        # Read file line by line to keep RAM usage minimal
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Tokenizing {os.path.basename(file_path)}"):
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    if not text:
                        continue

                    # Step 1: Tokenize the document into integer IDs
                    token_ids = sp.encode_as_ids(text)

                    # Step 2: Append the EOS token so the LLM knows this article is over
                    token_ids.append(EOS_ID)

                    # Step 3: Route to either Eval (5%) or Train (95%)
                    is_eval = random.random() < EVAL_RATIO

                    if is_eval:
                        eval_buffer.extend(token_ids)
                        eval_documents += 1

                        # Step 4a: Slice into exact block boundaries for Eval
                        while len(eval_buffer) >= SEQ_LEN:
                            block = eval_buffer[:SEQ_LEN]
                            eval_buffer = eval_buffer[SEQ_LEN:]

                            np_block = np.array(block, dtype=np.uint16)
                            f_eval.write(np_block.tobytes())
                            saved_eval_blocks += 1
                    else:
                        train_buffer.extend(token_ids)
                        train_documents += 1

                        # Step 4b: Slice into exact block boundaries for Train
                        while len(train_buffer) >= SEQ_LEN:
                            block = train_buffer[:SEQ_LEN]
                            train_buffer = train_buffer[SEQ_LEN:]

                            np_block = np.array(block, dtype=np.uint16)
                            f_train.write(np_block.tobytes())
                            saved_train_blocks += 1

                except json.JSONDecodeError:
                    continue

# Note: Any remaining tokens in the buffers at the very end of all files
# that do not sum to a full 1024-token block are discarded.
# This ensures 100% pure token chunks without any padding waste!

# ==========================================
# 4. Summary
# ==========================================
print("\n[Success] Tokenization, Splitting, and Packing Complete!")
print("-" * 50)
print("TRAINING DATA:")
print(f"-> Documents Processed: {train_documents}")
print(f"-> Blocks Created ({SEQ_LEN} tokens each): {saved_train_blocks}")
print(f"-> Total Tokens: {saved_train_blocks * SEQ_LEN:,}")
print(f"-> Saved to: {OUTPUT_TRAIN_BIN_FILE}")
print("-" * 50)
print("EVALUATION DATA:")
print(f"-> Documents Processed: {eval_documents}")
print(f"-> Blocks Created ({SEQ_LEN} tokens each): {saved_eval_blocks}")
print(f"-> Total Tokens: {saved_eval_blocks * SEQ_LEN:,}")
print(f"-> Saved to: {OUTPUT_EVAL_BIN_FILE}")
print("-" * 50)
