import json
import re
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset_cleaned.jsonl"



OUTPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset_filtered_v2.jsonl"

REVIEW_FILE = PROJECT_ROOT / "output" / "removed_records_review_v2.jsonl"

REPORT_FILE = PROJECT_ROOT / "output" / "final_filtering_report_v2.txt"
# =====================================================
# SETTINGS
# =====================================================

MIN_RESPONSE_LENGTH = 3

dictionary_abbreviations = [
    "adj.","adv.","v.","n.","s.",
    "pron.","prep.","conj.","interj.","aux.","cf."
]

# OCR garbage words observed in dataset
garbage_words = {
    "ಿ","ಾ","ು","ೂ","ೆ","ೇ","ೈ","ೊ","ೋ","ೌ",
    "ಓ","ಗೆ","ಗಿ","ತಿ","ಗ","ದ","ಕ್"
}

OCR_CORRECTIONS = {
    "ಪಾ ರುಪತ್ಯ್ಯ": "ಪಾರುಪತ್ಯ್ಯ",
    "ಪ್ರಮಾಣಪೂರ್ವಕಾ ದ್": "ಪ್ರಮಾಣಪೂರ್ವಕಾದ್",
}

# =====================================================
# CLEANING FUNCTIONS
# =====================================================

def collapse_repeated_punctuation(text):
    return re.sub(r"[.,;:!?]{2,}", lambda m: m.group(0)[0], text)


def remove_dictionary_abbreviations(text):
    for abbr in dictionary_abbreviations:
        text = text.replace(abbr, "")
    return text


def remove_latin_tokens(text):
    tokens = text.split()
    tokens = [
        token
        for token in tokens
        if not re.search(r"[A-Za-z]", token)
    ]
    return " ".join(tokens)


def normalize_spacing(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def apply_ocr_corrections(text):
    for wrong, correct in OCR_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def clean_response(text):

    text = str(text).strip()

    text = remove_dictionary_abbreviations(text)

    text = remove_latin_tokens(text)

    text = apply_ocr_corrections(text)

    text = collapse_repeated_punctuation(text)

    text = normalize_spacing(text)

    return text.strip()


def clean_instruction(text):

    text = str(text).strip()

    text = collapse_repeated_punctuation(text)

    text = normalize_spacing(text)

    return text


def is_garbage_response(text):

    compact = text.replace(" ", "")

    if len(compact) < MIN_RESPONSE_LENGTH:
        return True

    if compact in garbage_words:
        return True

    # only Kannada vowel signs
    if re.fullmatch(r'[\u0CCD-\u0CE3]+', compact):
        return True

    return False


# =====================================================
# LOAD DATA
# =====================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if line:

            records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# =====================================================
# FILTER
# =====================================================

filtered = []

removed = []

seen = set()

modified = 0

for item in records:

    original_instruction = item["instruction"]

    original_response = item["response"]

    instruction = clean_instruction(original_instruction)

    response = clean_response(original_response)

    if not instruction:

        removed.append({
            "instruction": original_instruction,
            "response": original_response,
            "reason": "Empty instruction"
        })

        continue

    if is_garbage_response(response):

        removed.append({
            "instruction": original_instruction,
            "response": original_response,
            "reason": "OCR garbage / too short"
        })

        continue

    key = (
        instruction.casefold(),
        response.casefold()
    )

    if key in seen:

        removed.append({
            "instruction": original_instruction,
            "response": original_response,
            "reason": "Duplicate"
        })

        continue

    seen.add(key)

    if response != original_response:

        modified += 1

    filtered.append({
        "instruction": instruction,
        "response": response
    })

# =====================================================
# SAVE
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for item in filtered:

        json.dump(item, f, ensure_ascii=False)

        f.write("\n")

with open(REVIEW_FILE, "w", encoding="utf-8") as f:

    for item in removed:

        json.dump(item, f, ensure_ascii=False)

        f.write("\n")

# =====================================================
# REPORT
# =====================================================

report = f"""
====================================================
FINAL FILTER REPORT
====================================================

Input Records     : {len(records)}

Filtered Records  : {len(filtered)}

Removed Records   : {len(removed)}

Modified Records  : {modified}

Output File

{OUTPUT_FILE}

Review File

{REVIEW_FILE}

====================================================
"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write(report)

print(report)