import json
import random
from pathlib import Path

# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"

INSTRUCTION_FILE = OUTPUT_DIR / "instruction_dataset.jsonl"
CONVERSATION_FILE = OUTPUT_DIR / "conversation_dataset.jsonl"

FINAL_FILE = OUTPUT_DIR / "final_training_dataset.jsonl"

# ======================================================
# Read JSONL Files
# ======================================================

dataset = []

for file_path in [INSTRUCTION_FILE, CONVERSATION_FILE]:

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            dataset.append(json.loads(line))

print(f"Total examples before shuffling: {len(dataset)}")

# ======================================================
# Shuffle Dataset
# ======================================================

random.shuffle(dataset)

print("Dataset shuffled successfully.")

# ======================================================
# Save Final Dataset
# ======================================================

with open(FINAL_FILE, "w", encoding="utf-8") as file:

    for item in dataset:

        json.dump(item, file, ensure_ascii=False)

        file.write("\n")

print(f"\nFinal training dataset saved to:\n{FINAL_FILE}")

# ======================================================
# Display Summary
# ======================================================

print(f"\nTotal Training Examples: {len(dataset)}")

print("\nFirst 5 Examples:\n")

for item in dataset[:5]:
    print(item)