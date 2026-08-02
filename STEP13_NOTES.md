# Step 13 — QLoRA Fine-tune Qwen3-4B

## Important limitation
This step's script was written and syntax-checked, but **could not be executed
in this sandbox** — the sandbox has no CUDA GPU and no internet access (needed
to download Qwen3-4B from Hugging Face the first time). QLoRA fine-tuning of a
4B-parameter model is not practical on CPU. You'll need to run
`scripts/finetune_qlora.py` yourself on a machine with an NVIDIA GPU (your own,
Google Colab, Kaggle, RunPod, Lambda, etc.).

## Script
`scripts/finetune_qlora.py`

Run it with:
```
python scripts/finetune_qlora.py
```

## What it does
1. Loads `Qwen/Qwen3-4B` in 4-bit precision (`bitsandbytes`, NF4, double
   quantization) so it fits on a consumer GPU — this is what makes it
   "QLoRA" rather than full fine-tuning.
2. Attaches LoRA adapters (rank 16, alpha 32, dropout 0.05) to all the
   attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`,
   `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Only these small adapter
   matrices are trained — the base model's weights stay frozen.
3. Loads `data/training/train.jsonl` and `data/training/validation.jsonl`
   (54,054 and 2,999 records respectively, from step 12).
4. Formats every `{instruction, response}` pair using Qwen's chat template
   (`tokenizer.apply_chat_template`), turning it into a proper
   user/assistant turn.
5. **Masks the loss on the prompt tokens** — the model is only trained to
   predict the Tulu response tokens, not to reproduce the instruction. This
   is standard practice for instruction fine-tuning and prevents the model
   from "wasting" capacity learning to copy the prompt.
6. Trains with the Hugging Face `Trainer`, evaluating on the validation set
   periodically, and keeps only the 3 best checkpoints
   (`load_best_model_at_end=True`, selected by lowest `eval_loss`).
7. Saves the final LoRA adapter to `models/qwen3-4b-tulu-lora/final_adapter/`
   and a `training_summary.json` with the run's key settings and final loss.

## Key hyperparameters (defaults, tune for your hardware)
| Setting | Default | Notes |
|---|---|---|
| Epochs | 3 | Reasonable starting point for ~60k examples |
| Per-device batch size | 4 | Lower if you get out-of-memory errors |
| Gradient accumulation | 4 | Effective batch size = 4 × 4 = 16 |
| Learning rate | 2e-4 | Typical for LoRA (higher than full fine-tuning) |
| LoRA rank (r) | 16 | Higher = more capacity, more VRAM/slower |
| Max sequence length | 512 | Dictionary entries are short; safe margin |

### Tuning for your GPU's VRAM
- **≤ 8 GB VRAM** (e.g. RTX 3060 Ti, RTX 4060): set
  `PER_DEVICE_BATCH_SIZE = 1` and `GRADIENT_ACCUMULATION_STEPS = 16`.
- **12–16 GB VRAM** (e.g. RTX 3060 12GB, RTX 4070 Ti, RTX 4080): defaults
  above should work.
- **24 GB+ VRAM** (e.g. RTX 3090/4090, A5000): you can raise
  `PER_DEVICE_BATCH_SIZE` to 8–16 and lower gradient accumulation
  accordingly, which will train faster.
- Always keep `per_device_batch_size × gradient_accumulation_steps` roughly
  constant if you want a similar effective batch size / training dynamics.

## New dependencies
`requirements.txt` was extended with: `torch`, `peft`, `bitsandbytes`,
`accelerate`, `datasets`, and the `transformers` version constraint was
raised to `>=4.51.0` (the first version with Qwen3 support).

## Bonus: quick sanity-check script
`scripts/quick_test_adapter.py` loads the trained adapter and runs a
handful of Tulu translation prompts through it, so you can eyeball whether
training worked before doing the full step 14 evaluation. Run it after
`finetune_qlora.py` finishes:
```
python scripts/quick_test_adapter.py
```

## Files added in step 13
| File | Purpose |
|---|---|
| `scripts/finetune_qlora.py` | Main QLoRA training script |
| `scripts/quick_test_adapter.py` | Post-training sanity check |
| `scripts/config.py` (modified) | Added `TRAINING_DATA` and `MODELS_DIR` paths |
| `requirements.txt` (modified) | Added training dependencies |
| `STEP13_NOTES.md` | This file |

## Next step (14)
Once you've run `finetune_qlora.py` and have a saved adapter, step 14 will
build a proper evaluation script — comparing the fine-tuned model's outputs
against the held-out `test.jsonl` (3,021 records) using metrics like BLEU/
character-level accuracy, not just eyeballing a few examples.
