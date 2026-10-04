import json
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer

# Import the model and configurations from your existing inference.py
from inference import GemmaConfig, GemmaForCausalLM


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt_tokens,
    max_new_tokens=128,
    temperature=0.3,
    top_p=0.9,
    device="cuda",
):
    """
    Generates a text response silently (without printing to console)
    and returns the fully generated string.
    """
    model.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_tokens = []
    repetition_penalty = 1.2

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.config.max_position_embeddings:
            break

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]

        # Apply Repetition Penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for token_id in set(generated_tokens):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

        # Sampling Mechanics
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        token_id = next_token.item()

        # Hard-Stopping Guardrails
        if (
            token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id]
            and len(generated_tokens) > 10
        ):
            break

        generated_tokens.append(token_id)
        full_text = tokenizer.decode(generated_tokens)

        if "។<bos>" in full_text or "។ <bos>" in full_text:
            break

        if token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_tokens)


def main():
    # --- Configuration Paths ---
    JSONL_PATH = "./eval/QA.jsonl"
    OUTPUT_TXT_PATH = "bayon_3.txt"
    CHECKPOINT_PATH = "checkpoints/bayon_it.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using device: {device}")

    # --- Load Tokenizer & Model ---
    print("[System] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("attentionlab/bayon")

    config = GemmaConfig()
    model = GemmaForCausalLM(config)

    print(f"[System] Loading model from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Could not find model checkpoint at: {CHECKPOINT_PATH}"
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"]
    clean_state_dict = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    print("[System] Model weights loaded successfully.")

    # --- Read QA dataset ---
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"Could not find input file: {JSONL_PATH}")

    questions = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Supports both "question" and "prompt" keys dynamically
                q = data.get("question") or data.get("prompt")
                if q:
                    questions.append(q)

    print(f"[System] Loaded {len(questions)} questions from {JSONL_PATH}")

    # --- Generation & Writing ---
    print(f"[System] Running benchmark. Writing outputs to {OUTPUT_TXT_PATH}...")

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as out_file:
        for idx, question in enumerate(tqdm(questions, desc="Benchmarking")):
            # Mirror the exact formatting used in your main CLI loop
            formatted_prompt = question + " || "
            prompt_tokens = tokenizer.encode(formatted_prompt)

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.9,
                device=device,
            )

            # Write structured format to output file
            out_file.write(f"--- Q&A Pair #{idx + 1} ---\n")
            out_file.write(f"Question: {question}\n")
            out_file.write(f"Response: {response.strip()}\n")
            out_file.write("=" * 60 + "\n\n")
            out_file.flush()  # Ensures progress is saved in real-time

    print(f"\n[System] Benchmark complete! Results saved to {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()
