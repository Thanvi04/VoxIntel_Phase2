import pandas as pd
from pathlib import Path
import json

# ======================================================
# File Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "final" / "final_dictionary.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
INSTRUCTION_DATASET_FILE = OUTPUT_DIR / "instruction_dataset.jsonl"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# Read Dictionary
# ======================================================

try:
    df = pd.read_excel(INPUT_FILE)
    print(" Dictionary loaded successfully.")
except FileNotFoundError:
    print(f" File not found:\n{INPUT_FILE}")
    exit()
except Exception as e:
    print(f" Error reading Excel file:\n{e}")
    exit()

# ======================================================
# Validate Required Columns
# ======================================================

required_columns = ["English", "Tulu"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

print(" Required columns found.")

# ======================================================
# Remove Empty Rows
# ======================================================

before_rows = len(df)

df = df.dropna(subset=["English", "Tulu"])

after_rows = len(df)

print(f"Removed {before_rows - after_rows} empty rows.")

# ======================================================
# Remove Duplicate English Words
# ======================================================

before_rows = len(df)

df = df.drop_duplicates(subset=["English"])

after_rows = len(df)

print(f"Removed {before_rows - after_rows} duplicate English words.")

# ======================================================
# Clean Whitespace
# ======================================================

df["English"] = df["English"].astype(str).str.strip()
df["Tulu"] = df["Tulu"].astype(str).str.strip()

print(" Whitespace cleaned.")

# ======================================================
# Display Information
# ======================================================

print("\nDictionary Summary")
print("------------------")
print(f"Total valid entries : {len(df)}")

print("\nSample Entries:\n")
print(df.head(10))

# ======================================================
# Save Validated Dictionary
# ======================================================

validated_file = OUTPUT_DIR / "validated_dictionary.xlsx"

df.to_excel(validated_file, index=False)

print(f"\n Validated dictionary saved to:\n{validated_file}")

# ======================================================
# Instruction Templates
# ======================================================

templates = [
    "Translate '{}' to Tulu.",
    "What is the Tulu meaning of {}?",
    "How do you say {} in Tulu?",
    "Give the Tulu translation of {}.",
    "Tulu word for {}."
]

print("\n Instruction templates created.")

# ======================================================
# Empty Dataset
# ======================================================

instruction_dataset = []

print(" Empty instruction dataset created.")

# ======================================================
# Loop Through Dictionary
# ======================================================

for index, row in df.iterrows():

    english = str(row["English"]).strip()

    tulu = str(row["Tulu"]).strip()

    meanings = [
        meaning.strip()
        for meaning in tulu.split(",")
        if meaning.strip()
    ]
    first_meaning = meanings[0]

    all_meanings = ", ".join(meanings)
    for i, template in enumerate(templates):

        instruction = template.format(english)

        if i == 1:
            response = all_meanings
        else:
            response = first_meaning

        instruction_dataset.append({

            "instruction": instruction,

            "response": response

        })   
print(f"\nGenerated {len(instruction_dataset)} instruction-response pairs.")
print("\nFirst 10 Examples:\n")

for item in instruction_dataset[:10]:

    print(item)

# ======================================================
# Save Instruction Dataset
# ======================================================

with open(INSTRUCTION_DATASET_FILE, "w", encoding="utf-8") as file:

    for item in instruction_dataset:

        json.dump(item, file, ensure_ascii=False)

        file.write("\n")

print(f"\n Instruction dataset saved to:\n{INSTRUCTION_DATASET_FILE}")