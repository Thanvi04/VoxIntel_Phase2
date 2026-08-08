"""
Step 13 - QLoRA fine-tuning of Qwen3-4B on the VoxIntel Tulu dataset.

IMPORTANT: This script needs a CUDA GPU (8 GB+ VRAM minimum for a 4B model
in 4-bit, 16 GB+ recommended for comfortable batch sizes) and an internet
connection the first time it runs (to download the base model from
Hugging Face). It will NOT run on a CPU-only machine in any reasonable
amount of time.

What this script does:
1. Loads Qwen3-4B in 4-bit (QLoRA) so it fits on a consumer GPU.
2. Attaches LoRA adapters to the attention/MLP projection layers.
3. Loads data/training/train.jsonl and data/training/validation.jsonl
   (produced by step 12) and formats each example using Qwen's chat
   template.
4. Masks the prompt tokens out of the loss, so the model is only trained
   to predict the Tulu response, not to reproduce the instruction.
5. Trains with the Hugging Face Trainer and saves the LoRA adapter
   (not a full merged model) to models/qwen3-4b-tulu-lora/.

Run from the project root:
    python scripts/finetune_qlora.py
"""

import json
import os

# Reduces CUDA memory fragmentation — set before torch/CUDA initializes.
# This is the fix PyTorch's own OOM error message recommends; without it,
# a GPU can report "out of memory" even when a good chunk of its total
# memory is technically free but too fragmented to serve one allocation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import TRAINING_DATA, MODELS_DIR


# ============================================================
# SETTINGS
# ============================================================

MODEL_ID = "Qwen/Qwen3-4B"

OUTPUT_DIR = MODELS_DIR / "qwen3-4b-tulu-lora"

TRAIN_FILE = TRAINING_DATA / "final_training_dataset_v3.jsonl"
VALIDATION_FILE = TRAINING_DATA / "final_validation_dataset_v3.jsonl"

MAX_SEQUENCE_LENGTH = 256

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters. UPDATED for Kaggle T4/P100 (16 GB VRAM,
# weaker compute than a 4090) running mostly-SHORT sequences (most
# VoxIntel examples are a few words in, a few words out). Batch
# size 1 (the original setting, tuned for long-sequence data on a
# faster GPU) wastes most of the GPU's parallelism on data this
# short. Start with these values; if you hit an out-of-memory error,
# halve PER_DEVICE_BATCH_SIZE and double GRADIENT_ACCUMULATION_STEPS
# to keep the same effective batch size of 32.
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8         # effective batch size = 32
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 25
EVAL_STEPS = 250
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 3
RANDOM_SEED = 42


# ============================================================
# CHECK INPUT FILES
# ============================================================

for path in (TRAIN_FILE, VALIDATION_FILE):

    if not path.exists():

        print("ERROR: Required training file not found:")
        print(path)

        print(
            "\nRun split_training_dataset.py first "
            "(step 8 or step 12)."
        )

        raise SystemExit(1)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK GPU
# ============================================================

if not torch.cuda.is_available():

    print("ERROR: No CUDA GPU detected.")

    print(
        "\nQLoRA fine-tuning of a 4B parameter model requires a "
        "CUDA-capable GPU. Training on CPU is not practical.\n"
        "\nOptions:\n"
        "  - Run this script on a machine with an NVIDIA GPU\n"
        "  - Use a cloud GPU (Google Colab, Kaggle, RunPod, Lambda, etc.)\n"
    )

    raise SystemExit(1)


gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print("=" * 70)
print("GPU DETECTED")
print("=" * 70)
print(f"Device : {gpu_name}")
print(f"Memory : {gpu_memory_gb:.1f} GB")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

# print(f"\nLoading tokenizer for {MODEL_ID} ...")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# tokenizer.padding_side = "right"


print(f"\nLoading tokenizer for {MODEL_ID} ...")

HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"



# ============================================================
# LOAD MODEL IN 4-BIT (QLoRA)
# ============================================================

print(f"\nLoading {MODEL_ID} in 4-bit ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=bnb_config,
#     device_map="auto",
# )
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)


# ============================================================
# ATTACH LORA ADAPTERS
# ============================================================

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading train / validation datasets ...")

raw_datasets = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VALIDATION_FILE),
    },
)

print(raw_datasets)


# ============================================================
# FORMAT + TOKENIZE
# ============================================================

def build_example(instruction, response):
    """
    Turns an (instruction, response) pair into Qwen chat-template
    text, and returns:
      - input_ids for the FULL sequence (prompt + response)
      - labels with the prompt portion masked out (-100), so the
        model is only trained to predict the response tokens.
    """

    prompt_messages = [
        {"role": "user", "content": instruction}
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )["input_ids"]

    labels = list(full_ids)

    prompt_length = min(len(prompt_ids), len(full_ids))

    for i in range(prompt_length):
        labels[i] = -100

    return full_ids, labels


def tokenize_function(examples):

    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for instruction, response in zip(
        examples["instruction"], examples["response"]
    ):

        input_ids, labels = build_example(instruction, response)

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_masks.append([1] * len(input_ids))

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }


print("\nTokenizing datasets ...")

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Tokenizing",
)

# ============================================================
# OPTIONAL: USE A SUBSET OF THE DATASET (set to None to use ALL data)
# ============================================================
#
# The full train set is ~61,813 examples / validation ~2,980.
# Whether that fits in one Kaggle GPU session depends on your
# measured steps/sec (see TRIAL_RUN below) — it is NOT capped here
# by default. If your trial-run estimate shows the full set won't
# fit in your session/quota, set TRAIN_SUBSET / VALIDATION_SUBSET
# to a number below instead of guessing blind.

TRAIN_SUBSET = None       # e.g. 30000 to cap it — None = use everything
VALIDATION_SUBSET = None  # e.g. 1500  to cap it — None = use everything

if TRAIN_SUBSET is not None:
    tokenized_datasets["train"] = tokenized_datasets["train"].select(
        range(min(TRAIN_SUBSET, len(tokenized_datasets["train"])))
    )

if VALIDATION_SUBSET is not None:
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
        range(min(VALIDATION_SUBSET, len(tokenized_datasets["validation"])))
    )

print(f"\nTraining samples: {len(tokenized_datasets['train'])}")
print(f"Validation samples: {len(tokenized_datasets['validation'])}")
# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================
# TRIAL RUN — measure real speed before committing a full session
# ============================================================
# Set TRIAL_RUN = True, run this script once, and read the printed
# estimate BEFORE launching the real 3-epoch job. This tells you
# whether the full run will fit in one Kaggle session (~9 hours)
# using YOUR actual batch size / GPU, not a guess.

TRIAL_RUN = True
TRIAL_MAX_STEPS = 200

import time


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=LOGGING_STEPS,
    eval_strategy="no" if TRIAL_RUN else "steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=False if TRIAL_RUN else True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=False,
    fp16=True,
    report_to="none",
    seed=RANDOM_SEED,
    max_steps=TRIAL_MAX_STEPS if TRIAL_RUN else -1,
)


# ============================================================
# TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

if TRIAL_RUN:
    print(f"\n=== TRIAL RUN: {TRIAL_MAX_STEPS} steps only ===\n")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    steps_per_sec = TRIAL_MAX_STEPS / elapsed
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps_full_run = (
        len(tokenized_datasets["train"]) // effective_batch * NUM_EPOCHS
    )
    est_hours = (total_steps_full_run / steps_per_sec) / 3600

    print(f"\nTrial: {TRIAL_MAX_STEPS} steps in {elapsed/60:.1f} min "
          f"-> {steps_per_sec:.3f} steps/sec")
    print(f"Full run: ~{total_steps_full_run} steps for {NUM_EPOCHS} epochs "
          f"over {len(tokenized_datasets['train'])} examples")
    print(f"ESTIMATED FULL TRAINING TIME: {est_hours:.1f} hours")
    print("\nKaggle free-GPU sessions cap out around 9 hours. If this "
          "estimate exceeds that:")
    print("  - lower TRAIN_SUBSET above instead of guessing")
    print("  - or lower NUM_EPOCHS to 2")
    print("  - or raise PER_DEVICE_BATCH_SIZE further if no OOM occurred")
    raise SystemExit(0)

print("\nStarting training ...\n")

train_result = trainer.train()

print("\nTraining complete.")
print(train_result)


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

final_adapter_dir = OUTPUT_DIR / "final_adapter"

model.save_pretrained(str(final_adapter_dir))
tokenizer.save_pretrained(str(final_adapter_dir))

print(f"\nFinal LoRA adapter saved to: {final_adapter_dir}")


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary = {
    "base_model": MODEL_ID,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_target_modules": LORA_TARGET_MODULES,
    "num_epochs": NUM_EPOCHS,
    "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "train_records": len(tokenized_datasets["train"]),
    "validation_records": len(tokenized_datasets["validation"]),
    "final_train_loss": train_result.training_loss,
    "adapter_path": str(final_adapter_dir),
}

summary_path = OUTPUT_DIR / "training_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)

print(f"Training summary saved to: {summary_path}")
