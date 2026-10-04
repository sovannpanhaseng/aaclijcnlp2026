import os

import torch
from transformers import GemmaConfig, GemmaForCausalLM


def convert_custom_pt_to_hf(checkpoint_path, output_dir="./bayon_hf_model"):
    print(f"[1/4] Loading local checkpoint from: {checkpoint_path}")
    # Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract the state dict (handles both raw weights or full training checkpoint dicts)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
    else:
        old_state_dict = checkpoint

    print("[2/4] Mapping custom keys to official Hugging Face Gemma structure...")
    new_state_dict = {}
    for key, value in old_state_dict.items():
        # Strip away torch.compile prefixes if they exist
        clean_key = key.replace("_orig_mod.", "")

        # In standard HF Gemma, the core layers live inside the 'model' attribute,
        # while the final prediction head lives on the root.
        if clean_key == "lm_head.weight":
            new_state_dict["lm_head.weight"] = value
        else:
            new_state_dict[f"model.{clean_key}"] = value

    print("[3/4] Initializing Hugging Face Gemma configuration...")
    # Perfectly match the hyperparameters from your GemmaConfig class
    hf_config = GemmaConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=28,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,  # hidden_size // num_attention_heads (512 // 8)
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        pad_token_id=0,
        bos_token_id=2,  # Standard defaults
        eos_token_id=1,
        torch_dtype="bfloat16",  # Force evaluation to bfloat16 to match your training autocast
    )

    # Initialize the official Hugging Face shell model
    hf_model = GemmaForCausalLM(hf_config)

    # Load your mapped weights into the shell
    hf_model.load_state_dict(new_state_dict, strict=True)
    print("[✔] Weight mapping successful and verified strict compliance!")

    print(f"[4/4] Saving production-ready artifacts to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # This automatically converts your .pt file into distributed .safetensors files
    # and generates your configuration JSONs seamlessly.
    hf_model.save_pretrained(output_dir, max_shard_size="5GB")
    print(
        "\n🎉 Conversion completed! Your model is optimized and ready for Hugging Face."
    )


if __name__ == "__main__":
    # Point this to your final training checkpoint asset
    CHECKPOINT = "checkpoints/bayon_it.pt"
    convert_custom_pt_to_hf(CHECKPOINT)
