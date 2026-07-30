import json
import random
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "final_training_dataset_cleaned.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "training"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"

VALIDATION_FILE = (
    OUTPUT_DIR
    / "validation.jsonl"
)

TEST_FILE = OUTPUT_DIR / "test.jsonl"

REPORT_FILE = (
    OUTPUT_DIR
    / "split_statistics.txt"
)


# ============================================================
# VALIDATE RATIOS
# ============================================================

total_ratio = (
    TRAIN_RATIO
    + VALIDATION_RATIO
    + TEST_RATIO
)

if abs(total_ratio - 1.0) > 1e-9:

    raise ValueError(
        "Train, validation and test ratios must add to 1.0"
    )


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():

    print("ERROR: Input dataset not found:")
    print(INPUT_FILE)

    print(
        "\nRun improve_training_dataset.py first."
    )

    raise SystemExit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

records = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(
        file,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        try:

            item = json.loads(line)

        except json.JSONDecodeError as error:

            print(
                f"Invalid JSON on line {line_number}: {error}"
            )

            raise SystemExit(1)


        instruction = str(
            item.get("instruction", "")
        ).strip()

        response = str(
            item.get("response", "")
        ).strip()


        if not instruction or not response:

            print(
                f"Empty instruction/response on line {line_number}"
            )

            raise SystemExit(1)


        records.append({
            "instruction": instruction,
            "response": response
        })


print(f"\nLoaded {len(records)} records.")


# ============================================================
# EXTRACT DICTIONARY WORD
# ============================================================

def extract_dictionary_word(instruction):

    """
    Detects the English dictionary word from the five
    instruction templates used in this project.

    Returns:
        English word if dictionary example
        None if conversation/other example
    """

    patterns = [

        r"^Translate ['\"](.+?)['\"] to Tulu\.$",

        r"^What is the Tulu meaning of (.+?)\?$",

        r"^How do you say (.+?) in Tulu\?$",

        r"^Give the Tulu translation of (.+?)\.$",

        r"^Tulu word for (.+?)\.$"
    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            instruction,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1).strip()


    return None


# ============================================================
# GROUP RELATED EXAMPLES
# ============================================================

dictionary_groups = defaultdict(list)

conversation_records = []


for item in records:

    word = extract_dictionary_word(
        item["instruction"]
    )

    if word:

        # casefold prevents Ability / ability
        # from becoming separate groups.

        dictionary_groups[
            word.casefold()
        ].append(item)

    else:

        conversation_records.append(item)


print(
    f"Dictionary word groups : {len(dictionary_groups)}"
)

print(
    f"Conversation/other examples : {len(conversation_records)}"
)


# ============================================================
# SHUFFLE GROUPS
# ============================================================

rng = random.Random(RANDOM_SEED)

dictionary_group_list = list(
    dictionary_groups.values()
)

rng.shuffle(dictionary_group_list)

rng.shuffle(conversation_records)


# ============================================================
# SPLIT DICTIONARY GROUPS
# ============================================================

number_of_groups = len(
    dictionary_group_list
)

train_group_end = int(
    number_of_groups
    * TRAIN_RATIO
)

validation_group_end = (
    train_group_end
    + int(
        number_of_groups
        * VALIDATION_RATIO
    )
)


train_dictionary_groups = (
    dictionary_group_list[
        :train_group_end
    ]
)

validation_dictionary_groups = (
    dictionary_group_list[
        train_group_end:
        validation_group_end
    ]
)

test_dictionary_groups = (
    dictionary_group_list[
        validation_group_end:
    ]
)


# ============================================================
# FLATTEN GROUPS
# ============================================================

def flatten(groups):

    result = []

    for group in groups:

        result.extend(group)

    return result


train_data = flatten(
    train_dictionary_groups
)

validation_data = flatten(
    validation_dictionary_groups
)

test_data = flatten(
    test_dictionary_groups
)


# ============================================================
# SPLIT CONVERSATION EXAMPLES
# ============================================================

conversation_count = len(
    conversation_records
)

conversation_train_end = int(
    conversation_count
    * TRAIN_RATIO
)

conversation_validation_end = (
    conversation_train_end
    + int(
        conversation_count
        * VALIDATION_RATIO
    )
)


train_conversations = (
    conversation_records[
        :conversation_train_end
    ]
)

validation_conversations = (
    conversation_records[
        conversation_train_end:
        conversation_validation_end
    ]
)

test_conversations = (
    conversation_records[
        conversation_validation_end:
    ]
)


# ============================================================
# ADD CONVERSATION DATA
# ============================================================

train_data.extend(
    train_conversations
)

validation_data.extend(
    validation_conversations
)

test_data.extend(
    test_conversations
)


# ============================================================
# SHUFFLE FINAL SPLITS
# ============================================================

rng.shuffle(train_data)
rng.shuffle(validation_data)
rng.shuffle(test_data)


# ============================================================
# SAFETY CHECK: EXACT DUPLICATE LEAKAGE
# ============================================================

def make_keys(dataset):

    return {
        (
            item["instruction"].casefold(),
            item["response"].casefold()
        )
        for item in dataset
    }


train_keys = make_keys(train_data)

validation_keys = make_keys(
    validation_data
)

test_keys = make_keys(test_data)


train_validation_overlap = (
    train_keys
    & validation_keys
)

train_test_overlap = (
    train_keys
    & test_keys
)

validation_test_overlap = (
    validation_keys
    & test_keys
)


if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):

    raise RuntimeError(
        "Data leakage detected between dataset splits."
    )


# ============================================================
# SAFETY CHECK: DICTIONARY WORD LEAKAGE
# ============================================================

def dictionary_words(dataset):

    words = set()

    for item in dataset:

        word = extract_dictionary_word(
            item["instruction"]
        )

        if word:

            words.add(
                word.casefold()
            )

    return words


train_words = dictionary_words(
    train_data
)

validation_words = dictionary_words(
    validation_data
)

test_words = dictionary_words(
    test_data
)


if train_words & validation_words:

    raise RuntimeError(
        "Dictionary word leakage between train and validation."
    )


if train_words & test_words:

    raise RuntimeError(
        "Dictionary word leakage between train and test."
    )


if validation_words & test_words:

    raise RuntimeError(
        "Dictionary word leakage between validation and test."
    )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in data:

            json.dump(
                item,
                file,
                ensure_ascii=False
            )

            file.write("\n")


save_jsonl(
    train_data,
    TRAIN_FILE
)

save_jsonl(
    validation_data,
    VALIDATION_FILE
)

save_jsonl(
    test_data,
    TEST_FILE
)


# ============================================================
# STATISTICS
# ============================================================

total = len(records)

actual_total = (
    len(train_data)
    + len(validation_data)
    + len(test_data)
)


statistics = f"""
============================================================
TRAIN / VALIDATION / TEST SPLIT
============================================================

Random seed:
{RANDOM_SEED}

Original dataset:
{total}

Dictionary groups:
{len(dictionary_groups)}

Conversation/other records:
{len(conversation_records)}

------------------------------------------------------------
TRAIN
------------------------------------------------------------

Examples:
{len(train_data)}

Percentage:
{len(train_data) / total * 100:.2f}%

Dictionary words:
{len(train_words)}

------------------------------------------------------------
VALIDATION
------------------------------------------------------------

Examples:
{len(validation_data)}

Percentage:
{len(validation_data) / total * 100:.2f}%

Dictionary words:
{len(validation_words)}

------------------------------------------------------------
TEST
------------------------------------------------------------

Examples:
{len(test_data)}

Percentage:
{len(test_data) / total * 100:.2f}%

Dictionary words:
{len(test_words)}

------------------------------------------------------------

Total after splitting:
{actual_total}

Exact train-validation overlap:
{len(train_validation_overlap)}

Exact train-test overlap:
{len(train_test_overlap)}

Exact validation-test overlap:
{len(validation_test_overlap)}

Dictionary word leakage:
0

============================================================
"""


# ============================================================
# SAVE STATISTICS
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(statistics)


# ============================================================
# PRINT RESULTS
# ============================================================

print(statistics)

print("Files saved successfully:\n")

print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)
print(REPORT_FILE)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nFirst 3 TRAIN examples:\n")

for item in train_data[:3]:

    print(item)


print("\nFirst 3 VALIDATION examples:\n")

for item in validation_data[:3]:

    print(item)


print("\nFirst 3 TEST examples:\n")

for item in test_data[:3]:

    print(item)