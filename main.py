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
from sympy.geometry.plane import t
from torch.amp import autocast
from torch.export.exported_program import PassType
from torch.fx.experimental.symbolic_shapes import safe_expand
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Graceful Exit Handler (Ctrl+C)
# ==========================================
exit_requested = False


def signal_handler(sig, frame):
    global exit_requested
    if not exit_requested:
        print(
            "\n[SIGINT detected] Graceful shutdown initiated. The script will save the model at the end of the current step and safely close."
        )
        exit_requested = True
    else:
        print("\n[Force Quit] Exiting immediately without saving.")
        sys.exit(1)


def ignore_sigint_in_worker(worker_id):
    """Prevents child processes from dying when Ctrl+C is pressed."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ==========================================
# 1.5. Deterministic Seeding
# ==========================================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[System] Global seed set to {seed}")


# ==========================================
# 2. Gemma Configuration & Binary Dataset
# ==========================================


class ResumableRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, items_to_skip=0, seed=42):
        self.dataset = dataset
        self.initial_skip = items_to_skip
        self.seed = seed
        self.epoch = 0
        self._is_first_epoch = True  # Use a flag instead

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


class GemmaConfig:
    def __init__(self):
        self.vocab_size = 5000  # Custom localized vocabulary
        self.hidden_size = 512  # Narrower embedding/hidden size
        self.intermediate_size = 2048  # GeGLU intermediate
        self.num_hidden_layers = 28  # Deeper architecture (28 layers)
        self.num_attention_heads = 8
        self.num_key_value_heads = 2  # Grouped Query Attention (GQA)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = 2048
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.pad_token_id = 0
        self.use_gradient_checkpointing = True  # Crucial for 6GB VRAM


class PackedBinaryDataset(Dataset):
    def __init__(self, bin_file, seq_len):
        self.seq_len = seq_len

        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"[Error] Binary data file not found: {bin_file}")

        # Read the binary file using memory mapping to save System RAM
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")

        # Calculate how many full sequences of seq_len we have
        self.num_sequences = len(self.data) // seq_len

        print(
            f"[System] Loaded dataset ({os.path.basename(bin_file)}): {self.num_sequences} sequences (Total tokens: {self.num_sequences * seq_len:,})"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Calculate the absolute start and end indices in the memory map
        start = idx * self.seq_len
        end = start + self.seq_len

        # Slice the numpy memory map and convert to a PyTorch tensor (int64)
        chunk = self.data[start:end]
        return torch.from_numpy(chunk.astype(np.int64))


# ==========================================
# 3. Model Architecture Components
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


def compute_rope_freqs(max_seq_len, dim, theta=10000.0, device="cuda"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs.to(device)).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(q, k, cos, sin, position_ids):
    cos_pos = cos[position_ids].unsqueeze(1)  # [seq_len, head_dim]
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

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids) * math.sqrt(
            self.config.hidden_size
        )
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        cos, sin = compute_rope_freqs(
            self.config.max_position_embeddings, self.config.head_dim, device=device
        )

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
# 4. Checkpointing & State Management
# ==========================================


def save_checkpoint(
    model, optimizer, scheduler, step, loss, filename="gemma_100m_ckpt"
):
    os.makedirs("checkpoints", exist_ok=True)
    filepath = f"checkpoints/{filename}_step_{step}.pt"
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state()
        if torch.cuda.is_available()
        else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(checkpoint_data, filepath)
    print(f"\n[System] Checkpoint saved successfully at {filepath}")


def load_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.isfile(filepath):
        print(f"[System] Loading checkpoint '{filepath}'...")
        ckpt = torch.load(filepath, map_location="cuda", weights_only=False)

        state_dict = ckpt["model_state_dict"]

        model_is_compiled = any(
            k.startswith("_orig_mod.") for k in model.state_dict().keys()
        )
        ckpt_is_compiled = any(k.startswith("_orig_mod.") for k in state_dict.keys())

        if model_is_compiled and not ckpt_is_compiled:
            state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        elif not model_is_compiled and ckpt_is_compiled:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

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
            f"[System] Resuming from step {ckpt['step']} with loss {ckpt['loss']:.4f}"
        )

        return ckpt["step"]
    print(f"[System] No checkpoint found at '{filepath}'. Starting from scratch.")
    return 0


# ==========================================
# 5. Validation Loop
# ==========================================


@torch.no_grad()
def evaluate(model, val_loader, config, device, eval_iters=50):
    model.eval()
    losses = []
    val_iter = iter(val_loader)

    for _ in range(eval_iters):
        try:
            inputs = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            inputs = next(val_iter)

        inputs = inputs.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
            )
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# ==========================================
# 6. Training Loop
# ==========================================


def train():
    avg_loss = 0.0

    global exit_requested
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = GemmaConfig()

    # --- Training Hyperparameters ---
    max_steps = 50000
    checkpoint_interval = 100
    log_interval = 25
    val_interval = 500
    batch_size = 4
    grad_accum_steps = 64
    seq_length = 1024

    # Track validation loss globally within train to prevent stale scoping issues
    val_loss = None

    # --- Load PRE-TOKENIZED Datasets ---
    data_file = "./data/pretrain_packed_data.bin"
    eval_data_file = "./data/eval_packed_data.bin"

    if not os.path.exists(data_file):
        print(
            f"[Error] Data file '{data_file}' not found. Please run the packing script first."
        )
        sys.exit(1)

    if not os.path.exists(eval_data_file):
        print(
            f"[Error] Evaluation data file '{eval_data_file}' not found. Ensure it exists with correct formatting."
        )
        sys.exit(1)

    train_dataset = PackedBinaryDataset(data_file, seq_length)
    val_dataset = PackedBinaryDataset(eval_data_file, seq_length)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=ignore_sigint_in_worker,
    )

    model = GemmaForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"[System] Initialized Gemma Model. Total Trainable Parameters: {total_params / 1e6:.2f}M"
    )

    print("[System] Compiling model for optimized execution...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(
            f"[Warning] torch.compile failed or not supported. Falling back to eager mode. ({e})"
        )

    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.1)

    # --- Scheduler Hyperparameters ---
    warmup_steps = 2500
    decay_end_step = 50000
    target_lr = 3e-5
    min_lr_ratio = target_lr / base_lr  # 0.1

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup from 0.0 to 1.0
            return float(current_step) / float(max(1, warmup_steps))
        if current_step > decay_end_step:
            # Hold minimum learning rate constant after step 50k
            return min_lr_ratio

        # Linear decay from 1.0 down to 0.1 between step 2500 and 50000
        progress = (current_step - warmup_steps) / (decay_end_step - warmup_steps)
        return 1.0 - progress * (1.0 - min_lr_ratio)

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_step = load_checkpoint(
        "checkpoints/latest_gemma_100m.pt", model, optimizer, scheduler
    )

    total_batches_to_skip = start_step * grad_accum_steps
    num_batches_per_epoch = len(train_dataset) // batch_size

    start_epoch = total_batches_to_skip // num_batches_per_epoch
    remaining_skip = total_batches_to_skip % num_batches_per_epoch

    # FIX: Convert remaining batch skips into exact token sequence skip items for the Sampler
    remaining_skip_items = remaining_skip * batch_size

    print(
        f"[System] Resuming at step {start_step}. Skipping {total_batches_to_skip} batches ({start_epoch} epochs + {remaining_skip} batches)."
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
    log_file = "training_stats.txt"

    print("\nStarting Training. Press Ctrl+C to safely save and exit at any time.")

    # Establish baseline time tracking for silent GPU hang
    with open("track_time.txt", "w") as file:
        file.write(str(time.time()))

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            try:
                inputs = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs = next(train_iter)

            inputs = inputs.to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
                )
                loss = loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Update time_tracker
        try:
            time_now = time.time()
            with open("track_time.txt", "w") as file:
                file.write(str(time_now))
            # if step % 2 == 0:
            #     print(f"[System] Last ping at Step {step},@ {time.strftime('%I:%M:%S %p', time.localtime(time_now))}")
            # not needed if using datacenter gpu
        except Exception as e:
            pass  # ignore I/O errors

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
            with open(log_file, "a") as f:
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
            with open(log_file, "a") as f:
                f.write(
                    f"[System] Running Validation at Step {step}...\n--> Validation Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}\n"
                )

        if step % checkpoint_interval == 0:
            # FIX: Use val_loss safely without relying on implicit scope evaluations
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                val_loss if val_loss is not None else avg_loss,
            )

            try:
                step_ckpt = f"checkpoints/gemma_100m_ckpt_step_{step}.pt"
                final_latest = "checkpoints/latest_gemma_100m.pt"

                os.replace(step_ckpt, final_latest)
            except Exception as e:
                print(f"[Warning] Failed to safely update latest checkpoint: {e}")
        if step % 500 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                val_loss if val_loss is not None else avg_loss,
            )
        if exit_requested:
            print("\n[System] Graceful exit triggered. Saving state...")
            steps_in_window = step % log_interval
            checkpoint_loss = (
                (total_loss / steps_in_window) if steps_in_window > 0 else total_loss
            )

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                checkpoint_loss,
                filename="safecancel_gemma_100m",
            )
            # FIX: High performance file copy optimization for graceful shutdowns
            try:
                temp_latest = "checkpoints/latest_gemma_100m.tmp"
                final_latest = "checkpoints/latest_gemma_100m.pt"
                safe_cancel_ckpt = f"checkpoints/safecancel_gemma_100m_step_{step}.pt"
                shutil.copy(safe_cancel_ckpt, temp_latest)
                os.replace(temp_latest, final_latest)
            except Exception as e:
                print(f"[Warning] Failed to map cancellation state link: {e}")
            print(
                "[System] Sleeping script for 3 seconds to ensure all IOPS are finished before exit."
            )
            time.sleep(3)
            sys.exit(0)

    print("[System] Pretraining completed.")
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        max_steps,
        total_loss / log_interval,
        filename="final_gemma_100m",
    )


if __name__ == "__main__":
    import os
    import subprocess
    import sys
    import time

    # ---------------------------------------------------------
    # WORKER MODE: This runs the actual PyTorch training
    # ---------------------------------------------------------
    if "--worker" in sys.argv:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        signal.signal(signal.SIGINT, signal_handler)
        train()

    # ---------------------------------------------------------
    # SUPERVISOR MODE: This watches the worker and restarts it
    # ---------------------------------------------------------
    else:
        print("[Watchdog] Starting training supervisor...")

        # How long to wait with NO updates before assuming the GPU hung
        # Note: If torch.compile takes longer than this on your machine, increase this value!
        TIMEOUT_SECONDS = 630  # 11 minutes

        while True:
            print("[Watchdog] Launching training worker...")

            # Reset heartbeat file so we don't immediately trigger a stale timeout
            with open("track_time.txt", "w") as f:
                f.write(str(time.time()))

            # Spawn the script itself as a subprocess worker
            worker = subprocess.Popen([sys.executable, __file__, "--worker"])

            try:
                while True:
                    # 1. Check if the worker crashed (driver crash, CUDA error, etc.)
                    retcode = worker.poll()
                    if retcode is not None:
                        if retcode == 0:
                            print("[Watchdog] Training completed successfully!")
                            sys.exit(0)
                        else:
                            print(
                                f"\n[Watchdog] Worker crashed (Exit Code {retcode}). Restarting in 15 seconds...\n"
                            )
                            time.sleep(15)
                            break  # Breaks the inner loop to restart the worker

                    # 2. Check for a silent hang (Utilization dropped to 0, no logs)
                    try:
                        if os.path.exists("track_time.txt"):
                            with open("track_time.txt", "r") as f:
                                content = f.read().strip()
                                if content:
                                    last_beat = float(content)
                                    if time.time() - last_beat > TIMEOUT_SECONDS:
                                        print(
                                            f"\n[Watchdog] ⚠️ Worker hung! No heartbeat for {TIMEOUT_SECONDS}s. Killing and restarting...\n"
                                        )
                                        worker.terminate()
                                        worker.wait()  # Wait for the OS to kill it
                                        torch.cuda.empty_cache()
                                        time.sleep(10)
                                        break
                    except (ValueError, IOError):
                        pass  # Ignore momentary read collisions

                    # Sleep briefly before checking again to save CPU
                    time.sleep(10)

            except KeyboardInterrupt:
                # If YOU press Ctrl+C, the Watchdog catches it, passes the signal to the child,
                # allows it to gracefully save, and then exits.
                print(
                    "\n[Watchdog] Supervisor (Watchdog) interrupted by user. Waiting for worker to gracefully shutdown..."
                )
                worker.wait()
                torch.cuda.empty_cache()
                sys.exit(0)
