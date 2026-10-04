"""
Generate on the 201 QA prompts with ANY Bayon checkpoint (e.g. the raw
pre-trained step-18k checkpoint, before SFT), using exactly the same
generation function and settings as bench.py, so LBR is comparable with your
existing SFT runs.  Output is in the same "--- Q&A Pair #N ---" text format,
so lbr.py reads it directly.

Put this file next to bench.py and inference.py.

  python bench_base.py --checkpoint checkpoints/<your_step18k>.pt --seed 0 --out base_pass1.txt
  python bench_base.py --checkpoint checkpoints/<your_step18k>.pt --seed 1 --out base_pass2.txt
  python bench_base.py --checkpoint checkpoints/<your_step18k>.pt --seed 2 --out base_pass3.txt

  python lbr.py base_pass1.txt base_pass2.txt base_pass3.txt --tokenizer attentionlab/bayon
  python lbr.py base_pass1.txt base_pass2.txt base_pass3.txt --char-level

The base model was never trained on the "question || answer" format, so it
will probably continue the text rather than answer.  That is fine for LBR
(it measures script of whatever is generated).  Report it as "LBR on
continuations", and report accuracy only if the outputs are meaningful.
Use --prompt-format "{q}" to feed the bare question instead.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from bench import generate_response  # same sampler, same settings
from inference import GemmaConfig, GemmaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--qa", default="./eval/QA.jsonl")
    ap.add_argument("--tokenizer", default="attentionlab/bayon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-format", default="{q} || ")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    model = GemmaForCausalLM(GemmaConfig())
    ckpt = torch.load(
        a.checkpoint,
        map_location=device,
        weights_only=False,
    )
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }
    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    questions = []
    with open(a.qa, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                q = d.get("question") or d.get("prompt")
                if q:
                    questions.append(q)
    print(f"{len(questions)} questions, seed {a.seed}, checkpoint {a.checkpoint}")

    with open(a.out, "w", encoding="utf-8") as out:
        for idx, q in enumerate(tqdm(questions)):
            prompt = a.prompt_format.format(q=q)
            resp = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=tokenizer.encode(prompt),
                max_new_tokens=a.max_new_tokens,
                temperature=a.temperature,
                top_p=a.top_p,
                device=device,
            )
            out.write(f"--- Q&A Pair #{idx + 1} ---\n")
            out.write(f"Question: {q}\n")
            out.write(f"Response: {resp.strip()}\n")
            out.write("=" * 60 + "\n\n")
            out.flush()


if __name__ == "__main__":
    main()
