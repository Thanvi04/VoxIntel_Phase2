import json
import re
from pathlib import Path
from collections import Counter

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset_cleaned.jsonl"

OUTPUT_FILE = PROJECT_ROOT / "output" / "final_training_dataset_filtered.jsonl"
REVIEW_FILE = PROJECT_ROOT / "output" / "removed_records_review.jsonl"
REPORT_FILE = PROJECT_ROOT / "output" / "final_filtering_report.txt"

# =====================================================
# SETTINGS
# =====================================================

# Minimum length a response must have AFTER cleaning to be kept.
MIN_RESPONSE_LENGTH = 2

# =====================================================
# LOAD DATA
# =====================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

print(f"\nLoaded {len(records)} records from:\n{INPUT_FILE}\n")

# =====================================================
# CLEANING HELPERS
# =====================================================

dictionary_abbreviations = [
    "adj.", "adv.", "v.", "n.", "s.",
    "pron.", "prep.", "conj.", "interj.", "aux.", "cf."
]


def collapse_repeated_punctuation(text):
    """
    Turn any run of 2+ punctuation marks (same or mixed, e.g. '..',
    ',,,', '.,.', ',,.') into a single punctuation mark (the first
    one in the run).
    """
    return re.sub(r"[.,;:!?]{2,}", lambda m: m.group(0)[0], text)


def remove_latin_tokens(text):
    """
    Remove whole tokens that contain any Latin/English letter.
    In this dataset, English fragments inside a Tulu/Kannada response
    are scan/OCR noise (dictionary abbreviations, stray headwords,
    leftover English words), not intended content.
    """
    tokens = text.split(" ")
    cleaned_tokens = [
        token for token in tokens
        if not re.search(r"[A-Za-z]", token)
    ]
    return " ".join(cleaned_tokens)


def remove_dictionary_abbreviations(text):
    for abbr in dictionary_abbreviations:
        text = text.replace(abbr, "")
    return text


def normalize_punctuation_spacing(text):
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Ensure a single space after punctuation (if followed by text)
    text = re.sub(r"([.,;:!?])(?=[^\s.,;:!?])", r"\1 ", text)
    return text


def strip_edge_punctuation(text):
    text = re.sub(r"^[.,;:\-\s]+", "", text)
    text = re.sub(r"[.,;:\-\s]+$", "", text)
    return text


def clean_response(text):
    text = str(text).strip()

    text = remove_dictionary_abbreviations(text)
    text = remove_latin_tokens(text)
    text = collapse_repeated_punctuation(text)
    text = normalize_punctuation_spacing(text)
    text = re.sub(r"\s+", " ", text)

    # Removing spaces before punctuation (normalize_punctuation_spacing)
    # can glue two separate marks back together (e.g. ". ." -> ".."),
    # so collapse repeated punctuation again as a final pass.
    text = collapse_repeated_punctuation(text)

    text = strip_edge_punctuation(text)

    return text.strip()


def clean_instruction(text):
    text = str(text).strip()
    text = collapse_repeated_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================================================
# FILTER DATASET
# =====================================================

filtered = []
removed = []
seen = set()

modified_count = 0
dropped_empty_or_short = 0
dropped_duplicate = 0

for item in records:

    original_instruction = item.get("instruction", "")
    original_response = item.get("response", "")

    instruction = clean_instruction(original_instruction)
    response = clean_response(original_response)

    # ---------------------------------------------
    # Drop records that no longer have a usable response
    # ---------------------------------------------
    if not instruction or not response or len(response) < MIN_RESPONSE_LENGTH:

        removed.append({
            "instruction": original_instruction,
            "response": original_response,
            "cleaned_response": response,
            "reason": "Empty or too short after cleaning"
        })

        dropped_empty_or_short += 1
        continue

    # ---------------------------------------------
    # Drop duplicates created by cleaning
    # ---------------------------------------------
    key = (instruction.casefold(), response.casefold())

    if key in seen:

        removed.append({
            "instruction": original_instruction,
            "response": original_response,
            "cleaned_response": response,
            "reason": "Duplicate after cleaning"
        })

        dropped_duplicate += 1
        continue

    seen.add(key)

    if response != str(original_response).strip():
        modified_count += 1

    filtered.append({
        "instruction": instruction,
        "response": response
    })

# =====================================================
# SAVE FILTERED DATASET
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for item in filtered:
        json.dump(item, file, ensure_ascii=False)
        file.write("\n")

# =====================================================
# SAVE REMOVED RECORDS FOR MANUAL REVIEW
# =====================================================

with open(REVIEW_FILE, "w", encoding="utf-8") as file:
    for item in removed:
        json.dump(item, file, ensure_ascii=False)
        file.write("\n")

# =====================================================
# RE-CHECK REMAINING SUSPICIOUS PATTERNS
# =====================================================

remaining_english = 0
remaining_repeated_punct = 0

repeated_punctuation_pattern = re.compile(r"([.,;:!?])\1{1,}")

for item in filtered:
    if re.search(r"[A-Za-z]", item["response"]):
        remaining_english += 1
    if repeated_punctuation_pattern.search(item["response"]):
        remaining_repeated_punct += 1

# =====================================================
# REPORT
# =====================================================

summary_lines = [
    "=" * 70,
    "STEP 11 - FINAL DATASET QUALITY FILTERING REPORT",
    "=" * 70,
    "",
    f"Input file  : {INPUT_FILE}",
    f"Output file : {OUTPUT_FILE}",
    f"Review file : {REVIEW_FILE}",
    "",
    "SUMMARY",
    "-" * 70,
    f"Records loaded                    : {len(records)}",
    f"Records kept (filtered dataset)   : {len(filtered)}",
    f"Records removed (empty/too short) : {dropped_empty_or_short}",
    f"Records removed (new duplicates)  : {dropped_duplicate}",
    f"Total removed                     : {len(removed)}",
    f"Records auto-fixed (text changed) : {modified_count}",
    "",
    "POST-FILTER QUALITY CHECK",
    "-" * 70,
    f"Responses still containing English/Latin chars : {remaining_english}",
    f"Responses still containing repeated punctuation : {remaining_repeated_punct}",
    "",
    "=" * 70,
]

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("\n".join(summary_lines))

print("\n".join(summary_lines))

print("\nFirst 5 removed records (manual review recommended):\n")
for item in removed[:5]:
    print("Instruction:", item["instruction"])
    print("Response   :", item["response"])
    print("Reason     :", item["reason"])
    print("-" * 50)

print(f"\nFiltered dataset saved to : {OUTPUT_FILE}")
print(f"Removed records saved to  : {REVIEW_FILE}")
print(f"Report saved to           : {REPORT_FILE}")
