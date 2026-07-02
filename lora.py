import datetime
import math
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 0. Note on loss masking
# ==========================================
# pack_sft.py now tokenizes the question and answer separately and emits a
# parallel *.mask.bin file (uint8, same block layout as the token .bin file)
# where 0 = question/prompt token, 1 = answer token or the trailing EOS. This
# script uses that mask so the SFT loss is computed only on answer tokens
# (i.e. the model isn't penalized for "predicting" the question, only for
# producing the right answer and knowing when to stop).

# ==========================================
# 1. Configuration
# ==========================================
BASE_MODEL_CHECKPOINT = (
    "checkpoints/mouy_100m.pt"  # frozen weights from main.py pretraining
)
SFT_TRAIN_BIN_FILE = "./data/sft_packed_train.bin"  # from pack_sft.py
SFT_EVAL_BIN_FILE = "./data/sft_packed_eval.bin"  # from pack_sft.py
SFT_TRAIN_MASK_FILE = "./data/sft_packed_train.mask.bin"  # from pack_sft.py
SFT_EVAL_MASK_FILE = "./data/sft_packed_eval.mask.bin"  # from pack_sft.py

LORA_CHECKPOINT_DIR = "checkpoints_sft_lora"
LATEST_LORA_CHECKPOINT = os.path.join(LORA_CHECKPOINT_DIR, "latest_lora_sft.pt")

HEARTBEAT_FILE = "track_time_sft.txt"
LOG_FILE = "sft_training_stats.txt"

# ==========================================
# 2. Graceful Exit Handler (Ctrl+C)
# ==========================================
exit_requested = False


def signal_handler(sig, frame):
    global exit_requested
    if not exit_requested:
        print(
            "\n[SIGINT detected] Graceful shutdown initiated. The script will save the LoRA adapter at the end of the current step and safely close."
        )
        exit_requested = True
    else:
        print("\n[Force Quit] Exiting immediately without saving.")
        sys.exit(1)


def ignore_sigint_in_worker(worker_id):
    """Prevents child processes from dying when Ctrl+C is pressed."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ==========================================
# 2.5 Deterministic Seeding
# ==========================================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[System] Global seed set to {seed}")


# ==========================================
# 3. Resumable Sampler & Packed Binary Dataset
# ==========================================


class ResumableRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, items_to_skip=0, seed=42):
        self.dataset = dataset
        self.initial_skip = items_to_skip
        self.seed = seed
        self.epoch = 0
        self._is_first_epoch = True

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if self._is_first_epoch and self.initial_skip > 0:
            indices = indices[self.initial_skip :]
            self._is_first_epoch = False

        self.epoch += 1
        return iter(indices)

    def __len__(self):
        if self._is_first_epoch:
            return len(self.dataset) - self.initial_skip
        return len(self.dataset)


class PackedSFTDataset(Dataset):
    """Reads fixed-length uint16 token blocks plus their parallel uint8 loss-mask
    blocks, both produced by pack_sft.py (0 = question/prompt token, 1 = answer
    token or trailing EOS). Open memmaps lazily to support safe worker spawning."""

    def __init__(self, bin_file, mask_file, seq_len):
        self.bin_file = bin_file
        self.mask_file = mask_file
        self.seq_len = seq_len

        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"[Error] Binary data file not found: {bin_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(
                f"[Error] Mask file not found: {mask_file}. Re-run pack_sft.py to generate it."
            )

        # Lazily opened arrays per process
        self.data = None
        self.mask = None

        # Size checking using metadata instead of opening full descriptors
        num_token_sequences = os.path.getsize(bin_file) // (
            2 * seq_len
        )  # uint16 = 2 bytes
        num_mask_sequences = os.path.getsize(mask_file) // (
            1 * seq_len
        )  # uint8 = 1 byte

        if num_token_sequences != num_mask_sequences:
            raise ValueError(
                f"[Error] Token/mask block count mismatch for '{bin_file}': "
                f"{num_token_sequences} token blocks vs {num_mask_sequences} mask blocks."
            )
        self.num_sequences = num_token_sequences

        print(
            f"[System] Loaded dataset ({os.path.basename(bin_file)}): {self.num_sequences} sequences "
            f"(Total tokens: {self.num_sequences * seq_len:,}, with answer-only loss mask)"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Lazy initialization happens inside the spawned process context
        if self.data is None:
            self.data = np.memmap(self.bin_file, dtype=np.uint16, mode="r")
        if self.mask is None:
            self.mask = np.memmap(self.mask_file, dtype=np.uint8, mode="r")

        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.data[start:end]
        mask_chunk = self.mask[start:end]
        return torch.from_numpy(chunk.astype(np.int64)), torch.from_numpy(
            mask_chunk.astype(np.float32)
        )


# ==========================================
# 4. Gemma Configuration
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
        self.use_gradient_checkpointing = True  # Crucial for low VRAM


# ==========================================
# 5. Model Architecture (identical to main.py)
# ==========================================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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

    q_embed = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (rotate_half(k) * sin_pos)
    return q_embed, k_embed


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
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        return self.o_proj(attn_output)


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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

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

        # Precompute RoPE once at startup to optimize execution loops
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

        # Reuse precomputed buffers sent dynamically to active runtime device
        cos = self.cos.to(device)
        sin = self.sin.to(device)

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, position_ids, cos, sin, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, position_ids, cos, sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ==========================================
# 6. LoRA
# ==========================================


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter."""

    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merged_weight(self):
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.base.weight + delta.to(self.base.weight.dtype)


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _set_module_by_name(root, dotted_name, new_module):
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_module)


def apply_lora_to_model(
    model, target_modules=DEFAULT_LORA_TARGET_MODULES, r=8, alpha=16, dropout=0.0
):
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in target_modules
        ):
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            _set_module_by_name(model, name, lora_layer)
            replaced += 1

    print(
        f"[System] Injected LoRA adapters into {replaced} linear layers (r={r}, alpha={alpha}, dropout={dropout})."
    )
    return model


def get_lora_state_dict(model):
    return {
        k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k
    }


def get_base_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def export_merged_model(model, config):
    raw = get_base_model(model)
    raw_sd = raw.state_dict()

    merged_sd = {}
    for name, module in raw.named_modules():
        if isinstance(module, LoRALinear):
            merged_sd[f"{name}.weight"] = module.merged_weight().clone()

    for k, v in raw_sd.items():
        if k.endswith(".base.weight") or ".lora_A" in k or ".lora_B" in k:
            continue
        if k not in merged_sd:
            merged_sd[k] = v.clone()

    merged = GemmaForCausalLM(config)
    merged.load_state_dict(merged_sd)
    merged.eval()
    return merged


# ==========================================
# 7. Checkpointing & State Management
# ==========================================


def save_lora_checkpoint(
    model, optimizer, scheduler, step, loss, checkpoint_dir, filename="lora_sft_ckpt"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"{filename}_step_{step}.pt")
    checkpoint_data = {
        "step": step,
        "lora_state_dict": get_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state()
        if torch.cuda.is_available()
        else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(checkpoint_data, filepath)
    print(f"\n[System] LoRA checkpoint saved successfully at {filepath}")
    return filepath


def load_base_checkpoint(filepath, model):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"[Error] Base pretrained checkpoint not found: {filepath}. Run main.py pretraining first."
        )

    print(f"[System] Loading pretrained base weights from '{filepath}'...")
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    model_is_compiled = any(
        k.startswith("_orig_mod.") for k in model.state_dict().keys()
    )
    ckpt_is_compiled = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if model_is_compiled and not ckpt_is_compiled:
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    elif not model_is_compiled and ckpt_is_compiled:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    print(
        f"[System] Base weights loaded (pretrained at step {ckpt.get('step', '?')}, "
        f"loss {ckpt.get('loss', float('nan')):.4f})."
    )


def load_lora_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.isfile(filepath):
        print(f"[System] Loading LoRA checkpoint '{filepath}'...")
        ckpt = torch.load(
            filepath,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )

        lora_sd = ckpt["lora_state_dict"]

        model_is_compiled = any(
            k.startswith("_orig_mod.") for k in model.state_dict().keys()
        )
        ckpt_is_compiled = any(k.startswith("_orig_mod.") for k in lora_sd.keys())

        if model_is_compiled and not ckpt_is_compiled:
            lora_sd = {f"_orig_mod.{k}": v for k, v in lora_sd.items()}
        elif not model_is_compiled and ckpt_is_compiled:
            lora_sd = {k.replace("_orig_mod.", ""): v for k, v in lora_sd.items()}

        model.load_state_dict(lora_sd, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if "torch_cuda_rng_state" in ckpt and ckpt["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])

        print(
            f"[System] Resuming SFT from step {ckpt['step']} with loss {ckpt['loss']:.4f}"
        )
        return ckpt["step"]

    print(
        f"[System] No LoRA checkpoint found at '{filepath}'. Starting LoRA adapter from scratch."
    )
    return 0


# ==========================================
# 8. Validation Loop
# ==========================================


def masked_next_token_loss(logits, input_ids, mask, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )
    mask_flat = shift_mask.view(-1)
    denom = mask_flat.sum().clamp(min=1.0)
    return (loss_flat * mask_flat).sum() / denom


@torch.no_grad()
def evaluate(model, val_loader, config, device, eval_iters=50):
    model.eval()
    losses = []
    val_iter = iter(val_loader)

    for _ in range(eval_iters):
        try:
            inputs, masks = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            inputs, masks = next(val_iter)

        inputs = inputs.to(device)
        masks = masks.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = masked_next_token_loss(logits, inputs, masks, config.vocab_size)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# ==========================================
# 9. LoRA SFT Training Loop
# ==========================================


def train():
    avg_loss = 0.0

    global exit_requested
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = GemmaConfig()

    # --- LoRA Hyperparameters ---
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = DEFAULT_LORA_TARGET_MODULES

    # --- SFT Training Hyperparameters ---
    num_epochs = 8
    checkpoint_interval = 50
    log_interval = 10
    val_interval = 50
    batch_size = 4
    grad_accum_steps = 16
    seq_length = 1024

    base_lr = 1e-4
    target_lr_ratio = 0.1
    warmup_ratio = 0.03

    val_loss = None

    for required_file in (
        SFT_TRAIN_BIN_FILE,
        SFT_TRAIN_MASK_FILE,
        SFT_EVAL_BIN_FILE,
        SFT_EVAL_MASK_FILE,
    ):
        if not os.path.exists(required_file):
            print(
                f"[Error] Required data file '{required_file}' not found. Please run pack_sft.py first."
            )
            sys.exit(1)

    train_dataset = PackedSFTDataset(
        SFT_TRAIN_BIN_FILE, SFT_TRAIN_MASK_FILE, seq_length
    )
    val_dataset = PackedSFTDataset(SFT_EVAL_BIN_FILE, SFT_EVAL_MASK_FILE, seq_length)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=ignore_sigint_in_worker,
    )

    # --- Build model: load frozen weights, then wrap target linear layers ---
    model = GemmaForCausalLM(config)
    load_base_checkpoint(BASE_MODEL_CHECKPOINT, model)
    apply_lora_to_model(
        model,
        target_modules=lora_target_modules,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[System] LoRA SFT model ready. Trainable params: {trainable_params / 1e6:.3f}M / "
        f"{total_params / 1e6:.2f}M total ({100 * trainable_params / total_params:.2f}%)"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=0.0)

    # --- Model Compilation (Executed AFTER optimizer tracking configuration) ---
    # print("[System] Compiling model for optimized execution...")
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(
    #         f"[Warning] torch.compile failed or not supported. Falling back to eager mode. ({e})"
    #     )

    num_batches_per_epoch = len(train_dataset) // batch_size
    if num_batches_per_epoch == 0:
        print(
            "[Error] Not enough packed sequences to fill a single batch. Reduce batch_size or add more SFT data."
        )
        sys.exit(1)

    max_steps = max(1, (num_batches_per_epoch * num_epochs) // grad_accum_steps)
    warmup_steps = max(1, int(warmup_ratio * max_steps))
    min_lr_ratio = target_lr_ratio

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step > max_steps:
            return min_lr_ratio
        progress = (current_step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return 1.0 - progress * (1.0 - min_lr_ratio)

    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs(LORA_CHECKPOINT_DIR, exist_ok=True)
    start_step = load_lora_checkpoint(
        LATEST_LORA_CHECKPOINT, model, optimizer, scheduler
    )

    print(
        f"[System] Plan: {num_epochs} epoch(s) over {len(train_dataset)} packed sequences "
        f"-> {max_steps} optimizer steps (grad_accum={grad_accum_steps}, batch_size={batch_size}, "
        f"warmup_steps={warmup_steps})."
    )

    total_batches_to_skip = start_step * grad_accum_steps
    start_epoch = total_batches_to_skip // num_batches_per_epoch
    remaining_skip = total_batches_to_skip % num_batches_per_epoch
    remaining_skip_items = remaining_skip * batch_size

    print(
        f"[System] Resuming at step {start_step}. Skipping {total_batches_to_skip} batches "
        f"({start_epoch} epochs + {remaining_skip} batches)."
    )

    train_sampler = ResumableRandomSampler(
        train_dataset, items_to_skip=remaining_skip_items, seed=42
    )
    train_sampler.epoch = start_epoch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        worker_init_fn=ignore_sigint_in_worker,
    )

    train_iter = iter(train_loader)

    model.train()
    start_time = time.time()
    total_loss = 0.0

    print("\nStarting LoRA SFT. Press Ctrl+C to safely save and exit at any time.")

    with open(HEARTBEAT_FILE, "w") as file:
        file.write(str(time.time()))

    if start_step >= max_steps:
        print(
            f"[System] start_step ({start_step}) >= max_steps ({max_steps}). Nothing left to train."
        )

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            try:
                inputs, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, masks = next(train_iter)

            inputs = inputs.to(device)
            masks = masks.to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = masked_next_token_loss(logits, inputs, masks, config.vocab_size)
                loss = loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        try:
            time_now = time.time()
            with open(HEARTBEAT_FILE, "w") as file:
                file.write(str(time_now))
        except Exception:
            pass

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed_time = time.time() - start_time
            steps_done = step - start_step
            avg_step_time = elapsed_time / steps_done

            remaining_steps = max_steps - step
            eta_seconds = int(remaining_steps * avg_step_time)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Step {step}/{max_steps} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Perplexity: {perplexity:.2f} | "
                f"Time/Step: {avg_step_time:.2f}s | "
                f"ETA: {eta_str}"
            )
            with open(LOG_FILE, "a") as f:
                f.write(
                    f"Step: {step}/{max_steps} | LR: {current_lr:.6f} | Train Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n"
                )
            total_loss = 0.0

        if step % val_interval == 0:
            print(f"[System] Running Validation at Step {step}...")
            val_loss = evaluate(model, val_loader, config, device)
            val_perplexity = math.exp(val_loss) if val_loss < 10 else float("inf")
            print(
                f"--> Validation Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}"
            )
            with open(LOG_FILE, "a") as f:
                f.write(
                    f"[System] Running Validation at Step {step}...\n--> Validation Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}\n"
                )

        if step % checkpoint_interval == 0:
            save_lora_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                val_loss if val_loss is not None else avg_loss,
                LORA_CHECKPOINT_DIR,
            )
            try:
                step_ckpt = os.path.join(
                    LORA_CHECKPOINT_DIR, f"lora_sft_ckpt_step_{step}.pt"
                )
                os.replace(step_ckpt, LATEST_LORA_CHECKPOINT)
            except Exception as e:
                print(f"[Warning] Failed to safely update latest LoRA checkpoint: {e}")

        if exit_requested:
            print("\n[System] Graceful exit triggered. Saving state...")
            steps_in_window = step % log_interval
            checkpoint_loss = (
                (total_loss / steps_in_window) if steps_in_window > 0 else total_loss
            )

            save_lora_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                checkpoint_loss,
                LORA_CHECKPOINT_DIR,
                filename="safecancel_lora_sft",
            )
            try:
                temp_latest = os.path.join(LORA_CHECKPOINT_DIR, "latest_lora_sft.tmp")
                safe_cancel_ckpt = os.path.join(
                    LORA_CHECKPOINT_DIR, f"safecancel_lora_sft_step_{step}.pt"
                )
                shutil.copy(safe_cancel_ckpt, temp_latest)
                os.replace(temp_latest, LATEST_LORA_CHECKPOINT)
            except Exception as e:
                print(f"[Warning] Failed to map cancellation state link: {e}")
            print(
                "[System] Sleeping script for 3 seconds to ensure all IOPS are finished before exit."
            )
            time.sleep(3)
            sys.exit(0)

    print("[System] LoRA SFT training completed.")
    final_path = save_lora_checkpoint(
        model,
        optimizer,
        scheduler,
        max_steps,
        val_loss if val_loss is not None else (total_loss / max(1, log_interval)),
        LORA_CHECKPOINT_DIR,
        filename="final_lora_sft",
    )
    try:
        temp_latest = os.path.join(LORA_CHECKPOINT_DIR, "latest_lora_sft.tmp")
        shutil.copy(final_path, temp_latest)
        os.replace(temp_latest, LATEST_LORA_CHECKPOINT)
    except Exception as e:
        print(
            f"[Warning] Failed to update latest LoRA checkpoint pointer after final save: {e}"
        )

    try:
        print(
            "[System] Merging LoRA adapter into base weights for standalone inference..."
        )
        merged = export_merged_model(model, config)
        merged_path = os.path.join(LORA_CHECKPOINT_DIR, "merged_final_model.pt")
        torch.save(
            {"step": max_steps, "model_state_dict": merged.state_dict()}, merged_path
        )
        print(f"[System] Merged inference-ready model saved at {merged_path}")
    except Exception as e:
        print(f"[Warning] Failed to export merged model: {e}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        signal.signal(signal.SIGINT, signal_handler)
        train()

    else:
        import subprocess

        print("[Watchdog] Starting LoRA SFT training supervisor...")

        TIMEOUT_SECONDS = 630

        while True:
            print("[Watchdog] Launching training worker...")

            with open(HEARTBEAT_FILE, "w") as f:
                f.write(str(time.time()))

            worker = subprocess.Popen([sys.executable, __file__, "--worker"])

            try:
                while True:
                    retcode = worker.poll()
                    if retcode is not None:
                        if retcode == 0:
                            print(
                                "[Watchdog] LoRA SFT training completed successfully!"
                            )
                            sys.exit(0)
                        else:
                            print(
                                f"\n[Watchdog] Worker crashed (Exit Code {retcode}). Restarting in 15 seconds...\n"
                            )
                            time.sleep(15)
                            break

                    try:
                        if os.path.exists(HEARTBEAT_FILE):
                            with open(HEARTBEAT_FILE, "r") as f:
                                content = f.read().strip()
                                if content:
                                    last_beat = float(content)
                                    if time.time() - last_beat > TIMEOUT_SECONDS:
                                        print(
                                            f"\n[Watchdog] ⚠️ Worker hung! No heartbeat for {TIMEOUT_SECONDS}s. Killing and restarting...\n"
                                        )
                                        worker.terminate()
                                        worker.wait()
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        time.sleep(10)
                                        break
                    except (ValueError, IOError):
                        pass

                    time.sleep(10)

            except KeyboardInterrupt:
                print(
                    "\n[Watchdog] Supervisor interrupted by user. Waiting for worker to gracefully shutdown..."
                )
                worker.wait()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sys.exit(0)
