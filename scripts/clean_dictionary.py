import pandas as pd
import re
from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

# Folder containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root (VoxIntel_Dataset_Cleaning)
PROJECT_DIR = SCRIPT_DIR.parent

# Data folders
RAW_DATA = PROJECT_DIR / "data" / "raw"
CLEAN_DATA = PROJECT_DIR / "data" / "cleaned"

# File paths
INPUT_FILE = RAW_DATA / "VoxIntel_dataset_Full_word.xlsx"
OUTPUT_FILE = CLEAN_DATA / "clean_dictionary.xlsx"

# =====================================================
# Check if file exists
# =====================================================

print("Input File:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found!\nExpected location:\n{INPUT_FILE}"
    )

# =====================================================
# Load Dataset
# =====================================================

print("\nLoading dataset...")

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()

print(df.columns.tolist())

print(f"Original Rows : {len(df)}")

# =====================================================
# Remove Empty Rows
# =====================================================

df.dropna(subset=["English", "Tulu"], inplace=True)

# =====================================================
# Remove Extra Spaces
# =====================================================

df["English"] = df["English"].astype(str).str.strip()
df["Tulu"] = df["Tulu"].astype(str).str.strip()

# =====================================================
# Remove Multiple Spaces
# =====================================================

df["English"] = df["English"].str.replace(r"\s+", " ", regex=True)
df["Tulu"] = df["Tulu"].str.replace(r"\s+", " ", regex=True)

# =====================================================
# Remove Hidden Unicode Characters
# =====================================================

def clean_unicode(text):

    text = str(text)

    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    return text.strip()

df["English"] = df["English"].apply(clean_unicode)
df["Tulu"] = df["Tulu"].apply(clean_unicode)

# =====================================================
# Remove Duplicate Rows
# =====================================================

df.drop_duplicates(inplace=True)

# =====================================================
# Remove Empty Strings
# =====================================================

df = df[
    (df["English"] != "") &
    (df["Tulu"] != "")
]

# =====================================================
# Reset Index
# =====================================================

df.reset_index(drop=True, inplace=True)

# =====================================================
# Save Dataset
# =====================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_excel(OUTPUT_FILE, index=False)

print("\n--------------------------------")
print("Cleaning Completed Successfully!")
print(f"Final Rows : {len(df)}")
print(f"Saved to : {OUTPUT_FILE}")