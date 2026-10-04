import json
import os
import re


def sanitize_khmer_text(text):
    """
    Remove characters that are not Khmer Unicode, spaces, question marks, or exclamation marks.
    Also removes all spaces before the firstKhmer character and all spaces after the last Khmer character.
    Returns the sanitized text.
    """
    if not text:
        return ""

    # Khmer Unicode ranges
    khmer_ranges = [
        (0x1780, 0x17FF),  # Khmer block
        (0x19E0, 0x19FF),  # Khmer Symbols
    ]

    allowed_chars = set(" ?!")
    result = []
    found_first_khmer = False

    for char in text:
        code = ord(char)
        is_khmer_char = any(start <= code <= end for start, end in khmer_ranges)

        if is_khmer_char:
            found_first_khmer = True
            result.append(char)
        elif char in allowed_chars:
            if char.isspace():
                # Only add spaces if we've already found a Khmer character
                if found_first_khmer:
                    result.append(" ")
            else:
                # Always add allowed punctuation
                result.append(char)

    # Remove all whitespace after the last Khmer character
    # Find the index of the last Khmer character
    last_khmer_idx = -1
    for i in range(len(result) - 1, -1, -1):
        char = result[i]
        code = ord(char)
        if any(start <= code <= end for start, end in khmer_ranges):
            last_khmer_idx = i
            break

    if last_khmer_idx != -1:
        # Remove any whitespace after the last Khmer character
        result = result[: last_khmer_idx + 1]

    sanitized_text = "".join(result)

    return sanitized_text


def extract_to_jsonl(input_dir=".", output_filename="input.jsonl"):
    # Clean regex: Matches 'gemini-code-' followed by any digits and ending in '.json'
    file_pattern = re.compile(r"^gemini-code-(\d+)\.json$")

    # Define your numeric boundaries safely here
    MIN_ID = 999999999999
    MAX_ID = 4999999999999

    item_count = 0
    file_count = 0
    duplicate_count = 0

    print(f"Scanning '{input_dir}' for matching Gemini JSON files...")

    # Load existing questions from output file if it exists
    seen_questions = set()
    if os.path.exists(output_filename):
        print(f"Loading existing questions from '{output_filename}'...")
        try:
            with open(output_filename, "r", encoding="utf-8") as existing_file:
                for line in existing_file:
                    if line.strip():
                        try:
                            existing_data = json.loads(line)
                            q = existing_data.get("question", "")
                            sanitized_q = sanitize_khmer_text(q)
                            seen_questions.add(sanitized_q)
                        except json.JSONDecodeError:
                            continue
            print(f"Loaded {len(seen_questions)} existing questions.")
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")

    # Open the output JSONL file in append mode
    with open(output_filename, "a", encoding="utf-8") as file:
        for filename in os.listdir(input_dir):
            match = file_pattern.match(filename)

            if match:
                # Extract the digits captured by (\d+) and convert to integer
                file_id = int(match.group(1))

                # Failsafe: Verify the ID falls strictly within your target range
                if not (MIN_ID <= file_id <= MAX_ID):
                    continue

                filepath = os.path.join(input_dir, filename)
                file_count += 1

                with open(filepath, "r", encoding="utf-8") as infile:
                    try:
                        data = json.load(infile)
                        items = data if isinstance(data, list) else [data]

                        for item in items:
                            question = item.get("question", "")
                            answer = item.get("answer", "")

                            # Sanitize the question
                            sanitized_question = sanitize_khmer_text(question)
                            if not sanitized_question:
                                continue

                            # sanitize answers
                            sanitized_answer = sanitize_khmer_text(answer)

                            # Check for duplicates
                            # if sanitized_question in seen_questions:
                            #     duplicate_count += 1
                            #     continue

                            consolidated_obj = {
                                "question": sanitized_question,
                                "answer": sanitized_answer,
                            }

                            # Write line to JSONL
                            file.write(
                                json.dumps(consolidated_obj, ensure_ascii=False) + "\n"
                            )
                            seen_questions.add(sanitized_question)
                            item_count += 1

                    except json.JSONDecodeError:
                        print(f"Skipping {filename}: Invalid JSON format.")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

    print("---")
    print("Extraction complete!")
    print(f"Processed {file_count} files containing {item_count} valid QA pairs.")
    print(f"Skipped {duplicate_count} duplicate questions.")
    print(f"Your data is ready in: {output_filename}")


if __name__ == "__main__":
    extract_to_jsonl()
