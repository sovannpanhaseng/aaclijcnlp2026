import json
import os

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
VOCAB_SIZE = 5000
INPUT_FILES = [
    "./data/corpus.txt"  # Can be a plain text file or a .jsonl file
]
TEMP_TEXT_FILE = "spm_training_data.txt"
MODEL_PREFIX = "tokenizerv1"

# Gemma Standard Special Tokens
USER_DEFINED_SYMBOLS = []  # Add specific Khmer punctuation if needed
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ==========================================
# 2. Prepare Raw Text for SentencePiece
# ==========================================


def prepare_training_text():
    print(f"[System] Extracting raw text from input files...")
    total_lines = 0

    # SentencePiece requires a raw text file where each line is a document/sentence
    with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as out_txt:
        for file_path in INPUT_FILES:
            if not os.path.exists(file_path):
                print(f"[Warning] {file_path} not found. Skipping.")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Processing {file_path}"):
                    line_str = line.strip()
                    if not line_str:
                        continue

                    # Hybrid parsing: Handles both plain text and JSONL automatically
                    try:
                        data = json.loads(line_str)
                        if isinstance(data, dict):
                            text = data.get("text", "").strip()
                        else:
                            text = line_str
                    except json.JSONDecodeError:
                        # Fallback if the file is just standard raw text
                        text = line_str

                    # Replace internal newlines with a space to help the tokenizer focus on tokens
                    text = text.replace("\n", " ").replace("\r", "")

                    if text:
                        out_txt.write(text + "\n")
                        total_lines += 1

    print(f"[System] Created temporary training file with {total_lines} documents.")


# ==========================================
# 3. Train SentencePiece Model
# ==========================================


def train_khmer_tokenizer():
    print(f"[System] Starting Tokenizer Training (Target Vocab: {VOCAB_SIZE})...")

    # SentencePiece Training Command
    spm.SentencePieceTrainer.train(
        input=TEMP_TEXT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",  # Gemma uses BPE
        character_coverage=0.9995,  # Capture almost all Khmer characters
        # CRITICAL: Forces non-Khmer text to break down into bytes
        byte_fallback=True,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=",".join(USER_DEFINED_SYMBOLS)
        if USER_DEFINED_SYMBOLS
        else "",
        max_sentence_length=16384,  # Handle long documents
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),  # Maximize CPU usage for speed
    )

    print(
        f"[Success] Tokenizer trained! Saved as {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab"
    )

    # Clean up the temporary text file
    if os.path.exists(TEMP_TEXT_FILE):
        os.remove(TEMP_TEXT_FILE)
        print("[System] Cleaned up temporary text file.")


# ==========================================
# 4. Quick Verification Test
# ==========================================


def test_tokenizer(model_path=None):
    model_path = model_path or f"{MODEL_PREFIX}.model"
    print(f"\n[System] Testing tokenizer: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test_khmer = "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ"  # "I love Khmer"
    test_english = "This English sentence is an afterthought."

    # Test Khmer (Should be dense, efficient tokens)
    khmer_tokens = sp.encode_as_pieces(test_khmer)
    khmer_ids = sp.encode_as_ids(test_khmer)
    print(f"\nKhmer Input: {test_khmer}")
    print(f"Pieces: {khmer_tokens}")
    print(f"IDs: {khmer_ids} (Token Count: {len(khmer_ids)})")

    # Test English (Should fragment heavily into bytes/individual letters)
    english_tokens = sp.encode_as_pieces(test_english)
    english_ids = sp.encode_as_ids(test_english)
    print(f"\nEnglish Input: {test_english}")
    print(f"Pieces: {english_tokens}")
    print(f"IDs: {english_ids} (Token Count: {len(english_ids)})")


if __name__ == "__main__":
    prepare_training_text()
    train_khmer_tokenizer()
    test_tokenizer()  # sanity check on the base tokenizer
