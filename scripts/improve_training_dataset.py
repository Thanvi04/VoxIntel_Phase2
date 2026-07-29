import json
import re
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset_cleaned.jsonl"

# =====================================================
# LOAD DATA
# =====================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))

print(f"\nLoaded {len(records)} records\n")

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = str(text).strip()

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated commas
    text = re.sub(r",+", ",", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


# =====================================================
# REMOVE DICTIONARY NOISE
# =====================================================

def remove_noise(text):

    text = str(text)

    # Dictionary abbreviations
    abbreviations = [
        "adj.",
        "adv.",
        "v.",
        "n.",
        "s.",
        "pron.",
        "prep.",
        "conj.",
        "interj.",
        "aux."
    ]

    for abbr in abbreviations:
        text = text.replace(abbr, "")

    # Remove isolated capital letters (T, A, B...)
    text = re.sub(r"\b[A-Z]\b", "", text)

    # Remove trailing English word
    text = re.sub(r"\s+[A-Za-z]+$", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading punctuation
    text = re.sub(r"^[.,;: ]+", "", text)

    # Remove trailing punctuation
    text = re.sub(r"[.,;: ]+$", "", text)

    return text.strip()

# =====================================================
# CLEAN DATASET
# =====================================================

cleaned = []
seen = set()

duplicates_removed = 0
modified_records = 0

print("=" * 60)
print("Modified Records")
print("=" * 60)

for item in records:

    instruction = clean_text(item["instruction"])

    original_response = item["response"]

    response = clean_text(original_response)
    response = remove_noise(response)

    if instruction == "" or response == "":
        continue

    key = (instruction, response)

    if key in seen:
        duplicates_removed += 1
        continue

    seen.add(key)

    cleaned.append({
        "instruction": instruction,
        "response": response
    })

    if response != original_response:

        modified_records += 1

        print("\nOLD :", original_response)
        print("NEW :", response)
        print("-" * 50)

# =====================================================
# SAVE CLEANED DATA
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for item in cleaned:

        json.dump(item, file, ensure_ascii=False)

        file.write("\n")

print(f"\nSaved {len(cleaned)} cleaned examples.")

# =====================================================
# SHOW FIRST 10
# =====================================================

print("\nFirst 10 cleaned examples:\n")

for i, item in enumerate(cleaned[:10], start=1):

    print(f"Example {i}")
    print("Instruction:", item["instruction"])
    print("Response   :", item["response"])
    print("-" * 50)

# =====================================================
# CHECK REMAINING ENGLISH WORDS
# =====================================================

print("\nChecking for remaining English words...\n")

count = 0

for item in cleaned:

    if re.search(r"[A-Za-z]", item["response"]):

        count += 1

        print(item["response"])

        if count == 20:
            break

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(f"Original records        : {len(records)}")
print(f"Final cleaned records   : {len(cleaned)}")
print(f"Duplicate records removed : {duplicates_removed}")
print(f"Modified responses      : {modified_records}")
print(f"Suspicious responses left (first 20 shown above): {count}")

print("=" * 60)