import pandas as pd
import json
from pathlib import Path

# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dataset response.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "conversation_dataset.jsonl"

print(INPUT_FILE)
print(INPUT_FILE.exists())
# ======================================================
# Read Excel
# ======================================================

df = pd.read_excel(INPUT_FILE)

print(f"Loaded {len(df)} conversation pairs.")

# ======================================================
# Keep Only Required Columns
# ======================================================

df = df[["tulu_text", "response_tulu"]]

# Rename columns
df = df.rename(columns={
    "tulu_text": "instruction",
    "response_tulu": "response"
})

# ======================================================
# Clean Data
# ======================================================

df = df.dropna(subset=["instruction", "response"])

df["instruction"] = df["instruction"].astype(str).str.strip()
df["response"] = df["response"].astype(str).str.strip()

df = df.drop_duplicates()

print(f"After cleaning: {len(df)} conversation pairs")

# ======================================================
# Save JSONL
# ======================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for _, row in df.iterrows():

        json.dump({
            "instruction": row["instruction"],
            "response": row["response"]
        }, file, ensure_ascii=False)

        file.write("\n")

print(f"\nConversation dataset saved to:\n{OUTPUT_FILE}")

# ======================================================
# Show Samples
# ======================================================

print("\nFirst 5 Examples:\n")

for i in range(min(5, len(df))):
    print(df.iloc[i].to_dict())