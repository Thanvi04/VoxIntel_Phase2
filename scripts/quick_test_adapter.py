"""
Quick sanity check for the LoRA adapter produced by finetune_qlora.py.

This is NOT the full step 14 evaluation (that comes later with proper
metrics). It just loads the base model + adapter and runs a handful of
Tulu prompts so you can eyeball whether training actually worked before
moving on.

Run from the project root, after finetune_qlora.py has finished:
    python scripts/quick_test_adapter.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import MODELS_DIR


MODEL_ID = "Qwen/Qwen3-4B"
ADAPTER_DIR = MODELS_DIR / "qwen3-4b-tulu-lora" / "final_adapter"

TEST_PROMPTS = [
    "Translate 'Water' to Tulu.",
    "How do you say Friend in Tulu?",
    "Tulu word for Beautiful.",
    "What is the Tulu meaning of House?",
]


if not ADAPTER_DIR.exists():

    print("ERROR: Adapter not found at:")
    print(ADAPTER_DIR)

    print("\nRun finetune_qlora.py first.")

    raise SystemExit(1)


if not torch.cuda.is_available():

    print("ERROR: No CUDA GPU detected. Cannot load a 4-bit model without one.")

    raise SystemExit(1)


print(f"Loading base model {MODEL_ID} in 4-bit ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

print(f"Loading LoRA adapter from {ADAPTER_DIR} ...")

model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()


print("\n" + "=" * 70)
print("QUICK SANITY CHECK")
print("=" * 70)

for prompt in TEST_PROMPTS:

    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nPrompt   : {prompt}")
    print(f"Response : {response.strip()}")
    print("-" * 50)

print("\nDone. Compare these against the expected Tulu translations manually.")
