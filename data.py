# import os
# import re
# import json
# import pyarrow.parquet as pq
# from tqdm import tqdm

# # ==========================================
# # 1. Configuration & Regex setup
# # ==========================================
# # Unicode ranges: Basic Khmer (1780-17FF) and Khmer Symbols (19E0-19FF)
# KHMER_REGEX = re.compile(r'[\u1780-\u17FF\u19E0-\u19FF]')

# MIN_LANGUAGE_SCORE = 0.90
# MIN_KHMER_RATIO = 0.85


# def passes_khmer_density_check(text):
#     if not text or not isinstance(text, str) or not text.strip():
#         return False

#     # Remove whitespaces and newlines to accurately calculate character ratio
#     text_no_spaces = text.replace(" ", "").replace(
#         "\n", "").replace("\r", "").replace("\t", "")
#     total_chars = len(text_no_spaces)

#     if total_chars == 0:
#         return False

#     khmer_chars = len(KHMER_REGEX.findall(text_no_spaces))
#     ratio = khmer_chars / total_chars

#     return ratio >= MIN_KHMER_RATIO

# # ==========================================
# # 2. Batch Processing Engine
# # ==========================================


# def process_single_parquet(parquet_path, out_file):
#     # Open the parquet file without loading the whole thing into memory
#     parquet_file = pq.ParquetFile(parquet_path)

#     accepted_count = 0
#     processed_count = 0

#     # iter_batches streams the file in chunks (e.g., 2000 rows at a time)
#     # This is crucial to prevent your 24GB RAM from overflowing
#     for batch in tqdm(parquet_file.iter_batches(batch_size=2000), desc=f"Reading {os.path.basename(parquet_path)}"):
#         # Convert the small batch to a pandas dataframe for easy row iteration
#         df_batch = batch.to_pandas()

#         for _, row in df_batch.iterrows():
#             processed_count += 1

#             # Extract fields (with fallbacks in case the column is missing/null)
#             lang_score = row.get("language_score", 0.0)
#             text = row.get("text", "")
#             original_id = row.get("id", "")

#             # 1. Language Score Filter
#             if lang_score < MIN_LANGUAGE_SCORE:
#                 continue

#             # 2. Unicode Density Filter
#             if passes_khmer_density_check(text):
#                 data_to_save = {
#                     "text": text,
#                     "source": os.path.basename(parquet_path),
#                     "original_id": original_id
#                 }
#                 out_file.write(json.dumps(
#                     data_to_save, ensure_ascii=False) + "\n")
#                 accepted_count += 1

#     return processed_count, accepted_count


# def process_directory(input_dir, output_filename):
#     print(f"\n[System] Looking for parquet files in: '{input_dir}'")

#     # Find all parquet files in the target directory
#     parquet_files = [os.path.join(input_dir, f) for f in os.listdir(
#         input_dir) if f.endswith(".parquet")]

#     if not parquet_files:
#         print(f"[Error] No .parquet files found in directory: {input_dir}")
#         return

#     print(
#         f"[System] Found {len(parquet_files)} parquet files. Starting extraction...")

#     total_processed = 0
#     total_accepted = 0

#     with open(output_filename, "w", encoding="utf-8") as out_file:
#         for p_file in parquet_files:
#             processed, accepted = process_single_parquet(p_file, out_file)
#             total_processed += processed
#             total_accepted += accepted

#     print(f"\n[Success] Direct Parquet processing complete.")
#     print(
#         f"-> Total Processed: {total_processed} | Total Accepted: {total_accepted}")
#     print(f"-> Saved highly filtered Khmer corpus to: {output_filename}")


# if __name__ == "__main__":
#     # 1. Put your downloaded FineWeb-2 removed parquet files into a folder named "parquet_data"
#     #    (or change this path to point exactly to where your files are downloaded)
#     INPUT_FOLDER = "./parquet_data"

#     # 2. The output JSONL file ready for the tokenizer/pretraining
#     OUTPUT_FILE = "khmer_corpus_recovered.jsonl"

#     # Create the input directory if it doesn't exist to prevent errors
#     os.makedirs(INPUT_FOLDER, exist_ok=True)

#     process_directory(
#         input_dir=INPUT_FOLDER,
#         output_filename=OUTPUT_FILE
#     )
# import pandas as pd
# import pyarrow.parquet as pq


# def extract_parquet_to_jsonl(parquet_file_path, output_jsonl_path, column_name="text", batch_size=20000):
#     try:
#         print(f"Opening {parquet_file_path}...")
#         # Open the Parquet file structure without loading the data into RAM
#         parquet_file = pq.ParquetFile(parquet_file_path)

#         # 1. Validate that the column exists before processing
#         if column_name not in parquet_file.schema.names:
#             print(
#                 f"Error: The column '{column_name}' was not found in the Parquet file.")
#             return

#         print(
#             f"Extracting '{column_name}' in batches to {output_jsonl_path}...")

#         # 2. Stream data in chunks and write directly to the JSONL file
#         with open(output_jsonl_path, 'w', encoding='utf-8') as f:
#             # iter_batches pulls a limited number of rows at a time into memory
#             for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[column_name]):
#                 # Convert only the current batch into a small Pandas DataFrame
#                 chunk_df = batch.to_pandas()

#                 # Convert the chunk into JSON lines format
#                 chunk_json = chunk_df.to_json(orient="records", lines=True)

#                 if chunk_json:
#                     f.write(chunk_json)
#                     # Ensure each chunk's ending transitions cleanly to the next line
#                     if not chunk_json.endswith('\n'):
#                         f.write('\n')

#         print("Extraction completed successfully!")

#     except Exception as e:
#         print(f"An error occurred: {e}")


# # Example Usage
# parquet_file = "./data/normal.parquet"
# jsonl_file = "khmer_corpus_normal.jsonl"

# # batch_size=20000 processes 20k rows at a time. Reduce this if your text rows are exceptionally long.
# extract_parquet_to_jsonl(parquet_file, jsonl_file,
#                          column_name="text", batch_size=40000)
import json


def combine_jsonl_text_only(file1_path, file2_path, output_txt_path, column_name="text"):
    try:
        print(f"Initializing output file: {output_txt_path}")

        # Open final TXT file for writing
        with open(output_txt_path, 'w', encoding='utf-8') as outfile:

            # 1. Process the first JSONL file
            print(f"Extracting '{column_name}' from {file1_path}...")
            with open(file1_path, 'r', encoding='utf-8') as infile1:
                for line_num, line in enumerate(infile1, 1):
                    if not line.strip():
                        continue  # Skip empty lines
                    try:
                        # json.loads automatically converts '\u179a' into real Khmer characters
                        data = json.loads(line)
                        text_content = data.get(column_name, "")

                        # Write only the text string to the file, followed by a newline
                        outfile.write(str(text_content) + '\n')
                    except json.JSONDecodeError:
                        # Gracefully skip if a line was cut in half during your preview test
                        print(
                            f"Warning: Skipped broken JSON on line {line_num} in {file1_path}")

            # 2. Process the second JSONL file
            print(f"Extracting '{column_name}' from {file2_path}...")
            with open(file2_path, 'r', encoding='utf-8') as infile2:
                for line_num, line in enumerate(infile2, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        text_content = data.get(column_name, "")
                        outfile.write(str(text_content) + '\n')
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Skipped broken JSON on line {line_num} in {file2_path}")

        print(f"Success! All raw text extracted into '{output_txt_path}'")

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the files. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example Usage
jsonl_part1 = "./khmer_corpus_normal.jsonl"
jsonl_part2 = "./data/khmer_corpus_recovered.jsonl"
combined_output_txt = "corpus.txt"

combine_jsonl_text_only(jsonl_part1, jsonl_part2,
                        combined_output_txt, column_name="text")
