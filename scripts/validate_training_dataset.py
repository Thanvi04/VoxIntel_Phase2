import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "final_training_dataset_filtered.jsonl"
)

REPORT_FILE = (
    BASE_DIR
    / "output"
    / "training_dataset_revalidation_report.txt"
)

SUSPICIOUS_FILE = (
    BASE_DIR
    / "output"
    / "suspicious_training_records_after_filtering.jsonl"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_INSTRUCTION_LENGTH = 2
MIN_RESPONSE_LENGTH = 1

MAX_INSTRUCTION_LENGTH = 500
MAX_RESPONSE_LENGTH = 1000


# ============================================================
# CHECK INPUT FILE
# ============================================================

print("=" * 70)
print("TRAINING DATASET VALIDATION")
print("=" * 70)

print(f"\nInput file:\n{INPUT_FILE}\n")

if not INPUT_FILE.exists():
    print("ERROR: Training dataset was not found.")
    print("\nExpected file:")
    print(INPUT_FILE)
    print("\nRun improve_training_dataset.py first.")
    raise SystemExit(1)


# ============================================================
# LOAD JSONL
# ============================================================

records = []

invalid_json_lines = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    for line_number, line in enumerate(file, start=1):

        line = line.strip()

        # Ignore completely empty lines
        if not line:
            continue

        try:

            item = json.loads(line)

            records.append({
                "_line_number": line_number,
                "_data": item
            })

        except json.JSONDecodeError as error:

            invalid_json_lines.append({
                "line": line_number,
                "error": str(error)
            })


print(f"Valid JSON records loaded : {len(records)}")
print(f"Invalid JSON lines        : {len(invalid_json_lines)}")


# ============================================================
# VALIDATION COUNTERS
# ============================================================

missing_instruction = []
missing_response = []

empty_instruction = []
empty_response = []

short_instruction = []
short_response = []

long_instruction = []
long_response = []

suspicious_responses = []

duplicate_records = []

seen = set()


# ============================================================
# SUSPICIOUS PATTERNS
# ============================================================

dictionary_abbreviation_pattern = re.compile(
    r"\b(?:adj|adv|pron|prep|conj|interj|aux|cf)\.",
    re.IGNORECASE
)

# English alphabet remaining inside the response.
# This is only a WARNING, not automatic deletion.
english_pattern = re.compile(r"[A-Za-z]")

# Repeated punctuation
repeated_punctuation_pattern = re.compile(
    r"([.,;:!?])\1{1,}"
)


# ============================================================
# VALIDATE EACH RECORD
# ============================================================

for record in records:

    line_number = record["_line_number"]
    item = record["_data"]

    # --------------------------------------------------------
    # Record must be a JSON object
    # --------------------------------------------------------

    if not isinstance(item, dict):

        suspicious_responses.append({
            "line": line_number,
            "reason": "Record is not a JSON object",
            "record": item
        })

        continue


    # --------------------------------------------------------
    # Check fields
    # --------------------------------------------------------

    if "instruction" not in item:

        missing_instruction.append(line_number)
        continue

    if "response" not in item:

        missing_response.append(line_number)
        continue


    instruction = item.get("instruction")
    response = item.get("response")


    # --------------------------------------------------------
    # Convert safely to strings
    # --------------------------------------------------------

    if instruction is None:
        instruction = ""

    if response is None:
        response = ""

    instruction = str(instruction).strip()
    response = str(response).strip()


    # --------------------------------------------------------
    # Empty values
    # --------------------------------------------------------

    if not instruction:

        empty_instruction.append(line_number)

    if not response:

        empty_response.append(line_number)


    # --------------------------------------------------------
    # Length checks
    # --------------------------------------------------------

    if instruction and len(instruction) < MIN_INSTRUCTION_LENGTH:

        short_instruction.append({
            "line": line_number,
            "instruction": instruction
        })


    if response and len(response) < MIN_RESPONSE_LENGTH:

        short_response.append({
            "line": line_number,
            "response": response
        })


    if len(instruction) > MAX_INSTRUCTION_LENGTH:

        long_instruction.append({
            "line": line_number,
            "length": len(instruction),
            "instruction": instruction
        })


    if len(response) > MAX_RESPONSE_LENGTH:

        long_response.append({
            "line": line_number,
            "length": len(response),
            "response": response
        })


    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    key = (
        instruction.casefold(),
        response.casefold()
    )

    if key in seen:

        duplicate_records.append({
            "line": line_number,
            "instruction": instruction,
            "response": response
        })

    else:

        seen.add(key)


    # --------------------------------------------------------
    # Suspicious response detection
    # --------------------------------------------------------

    reasons = []


    # English characters in Tulu response
    if english_pattern.search(response):

        reasons.append(
            "Contains English/Latin characters"
        )


    # Dictionary abbreviations
    if dictionary_abbreviation_pattern.search(response):

        reasons.append(
            "Contains dictionary abbreviation"
        )


    # Repeated punctuation
    if repeated_punctuation_pattern.search(response):

        reasons.append(
            "Contains repeated punctuation"
        )


    if reasons:

        suspicious_responses.append({
            "line": line_number,
            "instruction": instruction,
            "response": response,
            "reasons": reasons
        })


# ============================================================
# RESPONSE LENGTH STATISTICS
# ============================================================

response_lengths = []

instruction_lengths = []

for record in records:

    item = record["_data"]

    if not isinstance(item, dict):
        continue

    instruction = str(
        item.get("instruction", "")
    ).strip()

    response = str(
        item.get("response", "")
    ).strip()

    if instruction:
        instruction_lengths.append(len(instruction))

    if response:
        response_lengths.append(len(response))


def average(values):

    if not values:
        return 0

    return sum(values) / len(values)


# ============================================================
# SAVE SUSPICIOUS RECORDS
# ============================================================

SUSPICIOUS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    SUSPICIOUS_FILE,
    "w",
    encoding="utf-8"
) as file:

    for item in suspicious_responses:

        json.dump(
            item,
            file,
            ensure_ascii=False
        )

        file.write("\n")


# ============================================================
# CREATE REPORT
# ============================================================

summary_lines = [

    "=" * 70,
    "TRAINING DATASET VALIDATION REPORT",
    "=" * 70,

    "",

    f"Input file: {INPUT_FILE}",

    "",

    "DATASET SUMMARY",
    "-" * 70,

    f"Valid JSON records          : {len(records)}",
    f"Invalid JSON lines          : {len(invalid_json_lines)}",

    "",

    "FIELD VALIDATION",
    "-" * 70,

    f"Missing instruction         : {len(missing_instruction)}",
    f"Missing response            : {len(missing_response)}",
    f"Empty instruction           : {len(empty_instruction)}",
    f"Empty response              : {len(empty_response)}",

    "",

    "DUPLICATE VALIDATION",
    "-" * 70,

    f"Duplicate records           : {len(duplicate_records)}",

    "",

    "LENGTH VALIDATION",
    "-" * 70,

    f"Very short instructions     : {len(short_instruction)}",
    f"Very short responses        : {len(short_response)}",
    f"Very long instructions      : {len(long_instruction)}",
    f"Very long responses         : {len(long_response)}",

    "",

    f"Average instruction length  : {average(instruction_lengths):.2f}",
    f"Average response length     : {average(response_lengths):.2f}",

    "",

    "QUALITY WARNINGS",
    "-" * 70,

    f"Suspicious records          : {len(suspicious_responses)}",

    "",

    "NOTE:",
    "Suspicious records are warnings.",
    "They are NOT automatically removed.",

    "",

    f"Suspicious records file:",
    str(SUSPICIOUS_FILE),

    "",
    "=" * 70
]


# ============================================================
# DECIDE VALIDATION STATUS
# ============================================================

critical_errors = (
    len(invalid_json_lines)
    + len(missing_instruction)
    + len(missing_response)
    + len(empty_instruction)
    + len(empty_response)
    + len(duplicate_records)
)


if critical_errors == 0:

    validation_status = "PASSED"

else:

    validation_status = "FAILED"


summary_lines.append(
    f"VALIDATION STATUS: {validation_status}"
)

summary_lines.append(
    "=" * 70
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# ============================================================
# PRINT REPORT
# ============================================================

print()

for line in summary_lines:
    print(line)


# ============================================================
# PRINT SAMPLE SUSPICIOUS RECORDS
# ============================================================

if suspicious_responses:

    print("\nFirst 10 suspicious records:\n")

    for number, item in enumerate(
        suspicious_responses[:10],
        start=1
    ):

        print(f"Example {number}")

        if "instruction" in item:
            print(
                "Instruction:",
                item["instruction"]
            )

        if "response" in item:
            print(
                "Response   :",
                item["response"]
            )

        if "reasons" in item:
            print(
                "Reason     :",
                ", ".join(item["reasons"])
            )

        print("-" * 70)


print("\nValidation report saved to:")
print(REPORT_FILE)

print("\nSuspicious records saved to:")
print(SUSPICIOUS_FILE)