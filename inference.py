import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ==========================================
# 1. Model & Configuration Definitions
# (Must match your training script exactly)
# ==========================================


class GemmaConfig:
    def __init__(self):
        self.vocab_size = 5000
        self.hidden_size = 512
        self.intermediate_size = 2048
        self.num_hidden_layers = 28
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = 2048
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.pad_token_id = 0
        self.use_gradient_checkpointing = False  # Turned off for inference


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def compute_rope_freqs(max_seq_len, dim, theta=10000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(q, k, cos, sin, position_ids):
    cos_pos = cos[position_ids].unsqueeze(1)
    sin_pos = sin[position_ids].unsqueeze(1)
    cos_pos = torch.cat([cos_pos, cos_pos], dim=-1)
    sin_pos = torch.cat([sin_pos, sin_pos], dim=-1)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    return (q * cos_pos) + (rotate_half(q) * sin_pos), (k * cos_pos) + (
        rotate_half(k) * sin_pos
    )


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class GemmaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(self, hidden_states, position_ids, cos, sin):
        bsz, q_len, _ = hidden_states.size()
        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)
        k = torch.repeat_interleave(k, dim=1, repeats=self.num_kv_groups)
        v = torch.repeat_interleave(v, dim=1, repeats=self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(
            attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        )


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, position_ids, cos, sin):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids, cos, sin
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        cos, sin = compute_rope_freqs(
            config.max_position_embeddings, config.head_dim, device="cpu"
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        device = input_ids.device
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(
            self.config.hidden_size
        )
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        cos, sin = self.cos.to(device), self.sin.to(device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, cos, sin)

        return self.lm_head(self.norm(hidden_states))


# ==========================================
# 2. Token Streaming Generation Loop
# ==========================================


@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt_tokens,
    max_new_tokens=128,
    temperature=0.3,
    top_p=0.9,
    device="cuda",
):
    """
    Generates text token-by-token and prints it to the console immediately.
    """
    model.eval()

    # input_ids shape: (1, seq_len)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Store length tracking to cleanly decode byte sequences if necessary
    printed_text = ""
    generated_tokens = []

    # Optional: Set your repetition penalty factor here (1.0 = disabled, 1.15-1.2 = ideal)
    repetition_penalty = 1.2

    for _ in range(max_new_tokens):
        # Enforce maximum position capacity context constraints
        if input_ids.size(1) >= model.config.max_position_embeddings:
            break

        # Forward pass using precision matching training script (bfloat16)
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)
            # Pull only the last token position's distribution output: shape (1, vocab_size)
            next_token_logits = outputs[:, -1, :]

        # --- Apply Repetition Penalty ---
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            # next_token_logits is shape (1, vocab_size), extract the first batch row
            for token_id in set(generated_tokens):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

        # --- Sampling Mechanics ---
        if temperature > 0:
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # --- Top-P (Nucleus) Filtering ---
            if top_p > 0.0 and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)

                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens that fall outside the top_p threshold
                # We shift the mask to keep the first token that exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the mask back to the original logits shape and mask out unwanted tokens
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append step token to running matrix array sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        token_id = next_token.item()

        # --- Mandatory Hard-Stopping Guardrails ---
        # 1. Stop if it emits a rogue <bos> or <eos> token after generating text
        if (
            token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id]
            and len(generated_tokens) > 10
        ):
            break

        generated_tokens.append(token_id)

        # Decode the complete running delta chunk to properly account for multi-byte tokens
        full_text = tokenizer.decode(generated_tokens)
        new_text = full_text[len(printed_text) :]
        printed_text = full_text

        # 2. String-level intercept to kill the "AI boilerplate" or "construction loop" instantly
        if "។<bos>" in full_text or "។ <bos>" in full_text:
            # Erase the trailing broken loop phrase from printing if desired, then kill stream
            break

        # Stream the token output to the console live
        sys.stdout.write(new_text)
        sys.stdout.flush()
        # Break loop execution context if your model meets its EOS terminator signal
        # (Assuming your tokenizer has an eos_token_id attribute, change to specific ID if needed)
        if token_id == tokenizer.eos_token_id:
            break

    print()  # Final newline wrapping statement execution loop


# ==========================================
# 3. Execution Wrapper Entry Point
# ==========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path configuration matching your export targets
    CHECKPOINT_PATH = "checkpoints/bayon_it.pt"

    # --- Tokenizer Note ---
    # Since your script has a fixed custom vocab_size of 5000, initialize the exact
    # tokenizer instance wrapper script you used within pack_sft.py here.
    # Replace this placeholder with your exact tokenizer setup:
    from transformers import AutoTokenizer

    print("[System] Loading tokenizer configuration mapping...")
    tokenizer = AutoTokenizer.from_pretrained("attentionlab/bayon")

    # Initialize raw configuration architecture block parameters
    config = GemmaConfig()
    model = GemmaForCausalLM(config)

    print(f"[System] Loading model parameters from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Could not find model checkpoint at: {CHECKPOINT_PATH}"
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    # Strip any potential compiled module prefix tags matching your training configuration rules
    state_dict = ckpt["model_state_dict"]
    clean_state_dict = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    print("[System] Model weights mapped successfully.")

    # Context Prompt Interface Input Loop
    print("\n" + "=" * 50)
    print("Gemma Custom Streaming Inference Shell (Type 'exit' to quit)")
    print("=" * 50 + "\n")

    while True:
        try:
            user_prompt = input("Prompt >>> ")
            if user_prompt.strip().lower() == "exit":
                break
            if not user_prompt.strip():
                continue

            formatted_prompt = user_prompt + " || "
            # Process prompt down into token ids array matrix indices
            prompt_tokens = tokenizer.encode(formatted_prompt)

            print("Response: ", end="")
            generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.9,
                device=device,
            )
            print("\n" + "-" * 30)
        except KeyboardInterrupt:
            print("\n[System] Session canceled.")
            break


if __name__ == "__main__":
    main()
