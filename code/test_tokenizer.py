import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import tiktoken
from openai import APIError, OpenAI, RateLimitError
from transformers import AutoTokenizer

# ==============================================================================
# CONFIGURATION & API CLIENT SETUP
# ==============================================================================
GROQ_API_KEY = ""
GROQ_BASE_URL = "https://api.cerebras.ai/v1/"
GPT_MODEL_NAME = "gpt-oss-120b"

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)

SEALION_API_KEY = ""
SEALION_BASE_URL = "https://api.sea-lion.ai/v1"
QWEN_MODEL_NAME = "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"

sealion_client = OpenAI(
    api_key=SEALION_API_KEY,
    base_url=SEALION_BASE_URL,
)

GENERATION_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.4,
    "top_p": 0.9,
}

CALL_TIMEOUT_SECONDS = 180  # auto-skip if no response within 3 minutes

print("[Info] Loading tokenizers...")
o200k_enc = tiktoken.get_encoding("o200k_base")

qwen_tokenizer = AutoTokenizer.from_pretrained(
    "aisingapore/Qwen-SEA-LION-v4.5-27B-IT", trust_remote_code=True
)


def extract_questions(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        questions = re.findall(
            r"(?:Question|Prompt):\s*(.*?)(?=\n(?:Response|Question|Prompt|\=)|\Z)",
            content,
            re.DOTALL,
        )

        if not questions:
            questions = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.startswith("=")
            ]

        return [q.strip() for q in questions if q.strip()]
    except FileNotFoundError:
        print(f"[Error] File not found: {filename}")
        return []


def call_api_with_retry(client, model_name, messages, max_retries=3):
    delay = 15.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, **GENERATION_PARAMS
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(
                f"  [Rate Limit] Hit rate limit on {model_name}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2
        except APIError as e:
            print(f"  [API Error] {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"  [Unexpected Error] {e}")
            break
    return None


def call_with_timeout(client, model_name, messages, timeout=CALL_TIMEOUT_SECONDS):
    """Runs call_api_with_retry in worker thread, gives up after timeout seconds.
    Does not wait for the abandoned thread to finish (non-blocking shutdown)."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call_api_with_retry, client, model_name, messages)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        print(f"  [Timeout] {model_name} exceeded {timeout}s. Skipping.")
        return None
    finally:
        executor.shutdown(wait=False)


def analyze_token_string(token_str):
    cleaned = token_str.replace(" ", "").replace("Ġ", "").replace(" ", "")

    is_byte_fallback = bool(re.match(r"^<0x[0-9A-Fa-f]{2}>$", cleaned))

    has_foreign_script = bool(re.search(r"[a-zA-Z\u0E00-\u0E7F\u4E00-\u9FFF]", cleaned))

    return is_byte_fallback, has_foreign_script


def evaluate_gpt_o200k(text):
    token_ids = o200k_enc.encode(text)
    total_tokens = len(token_ids)
    bleed_tokens = 0
    fallback_tokens = 0

    for token_id in token_ids:
        raw_bytes = o200k_enc.decode_single_token_bytes(token_id)
        try:
            token_str = raw_bytes.decode("utf-8")
            is_fallback, is_foreign = analyze_token_string(token_str)
        except UnicodeDecodeError:
            is_fallback = True
            is_foreign = False

        if is_fallback:
            fallback_tokens += 1
        if is_foreign or is_fallback:
            bleed_tokens += 1

    return total_tokens, bleed_tokens, fallback_tokens


def evaluate_qwen_tokenizer(text):
    token_ids = qwen_tokenizer.encode(text)
    token_strings = qwen_tokenizer.convert_ids_to_tokens(token_ids)

    total_tokens = len(token_ids)
    bleed_tokens = 0
    fallback_tokens = 0

    for token_str in token_strings:
        is_fallback, is_foreign = analyze_token_string(token_str)

        if is_fallback:
            fallback_tokens += 1
        if is_foreign or is_fallback:
            bleed_tokens += 1

    return total_tokens, bleed_tokens, fallback_tokens


def main():
    input_file = "bayon_1.txt"
    questions = extract_questions(input_file)

    if not questions:
        print(f"[Warning] No questions found in {input_file}. Exiting.")
        return

    print(f"[Info] Processing {len(questions)} question(s) from {input_file}...\n")

    gpt_total, gpt_bleed, gpt_fallback = 0, 0, 0
    qwen_total, qwen_bleed, qwen_fallback = 0, 0, 0
    gpt_skipped, qwen_skipped = 0, 0

    for idx, question in enumerate(questions, 1):
        print(f"--- Prompt {idx}/{len(questions)} ---")
        messages = [{"role": "user", "content": question}]

        print(f"  [Call] {GPT_MODEL_NAME} ...", flush=True)
        t0 = time.time()
        gpt_text = call_with_timeout(groq_client, GPT_MODEL_NAME, messages)
        if gpt_text:
            print(
                f"  [OK] {GPT_MODEL_NAME} responded in {time.time() - t0:.1f}s ({len(gpt_text)} chars)"
            )
            g_tot, g_bld, g_fb = evaluate_gpt_o200k(gpt_text)
            gpt_total += g_tot
            gpt_bleed += g_bld
            gpt_fallback += g_fb
        else:
            gpt_skipped += 1
            print(
                f"  [Skip] {GPT_MODEL_NAME} returned nothing for prompt {idx} after {time.time() - t0:.1f}s. Skipping."
            )

        print(f"  [Call] {QWEN_MODEL_NAME} ...", flush=True)
        t0 = time.time()
        qwen_text = call_with_timeout(sealion_client, QWEN_MODEL_NAME, messages)
        if qwen_text:
            print(
                f"  [OK] {QWEN_MODEL_NAME} responded in {time.time() - t0:.1f}s ({len(qwen_text)} chars)"
            )
            q_tot, q_bld, q_fb = evaluate_qwen_tokenizer(qwen_text)
            qwen_total += q_tot
            qwen_bleed += q_bld
            qwen_fallback += q_fb
        else:
            qwen_skipped += 1
            print(
                f"  [Skip] {QWEN_MODEL_NAME} returned nothing for prompt {idx} after {time.time() - t0:.1f}s. Skipping."
            )

        time.sleep(1.0)

    print("\n" + "=" * 65)
    print("         LANGUAGE BLEED RATIO (LBR) REPORT")
    print("=" * 65)

    print(f"\n[Model: {GPT_MODEL_NAME} | Tokenizer: o200k_base]")
    print(f"  - Skipped (empty response) : {gpt_skipped}/{len(questions)}")
    print(f"  - Total Tokens Counted     : {gpt_total:,}")
    print(f"  - Byte-Fallback Tokens     : {gpt_fallback:,}")
    print(f"  - Total Bleed Tokens       : {gpt_bleed:,}")
    gpt_lbr = (gpt_bleed / gpt_total * 100) if gpt_total > 0 else 0.0
    print(f"  - Language Bleed Ratio     : {gpt_lbr:.4f}%")

    print(f"\n[Model: {QWEN_MODEL_NAME} | Tokenizer: Qwen 2.5 / 3.5]")
    print(f"  - Skipped (empty response) : {qwen_skipped}/{len(questions)}")
    print(f"  - Total Tokens Counted     : {qwen_total:,}")
    print(f"  - Byte-Fallback Tokens     : {qwen_fallback:,}")
    print(f"  - Total Bleed Tokens       : {qwen_bleed:,}")
    qwen_lbr = (qwen_bleed / qwen_total * 100) if qwen_total > 0 else 0.0
    print(f"  - Language Bleed Ratio     : {qwen_lbr:.4f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
