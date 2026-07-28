import pandas as pd
import re

from config import CLEANED_DATA, FINAL_DATA, OUTPUT_DATA

# =====================================================
# File Paths
# =====================================================

INPUT_FILE = CLEANED_DATA / "clean_dictionary.xlsx"
REPORT_FILE = OUTPUT_DATA / "validation_report.xlsx"
OUTPUT_FILE = FINAL_DATA / "final_dictionary.xlsx"

# =====================================================
# Load Files
# =====================================================

print("Loading cleaned dictionary...")

dictionary = pd.read_excel(INPUT_FILE)
report = pd.read_excel(REPORT_FILE)

dictionary.columns = dictionary.columns.str.strip()
report.columns = report.columns.str.strip()

print(f"Dictionary Rows : {len(dictionary)}")
print(f"Issues Found    : {len(report)}")

# =====================================================
# Fix Reported Issues
# =====================================================

for _, issue in report.iterrows():

    row = int(issue["Row"]) - 2

    if row < 0 or row >= len(dictionary):
        continue

    severity = issue["Severity"]
    problem = issue["Issue"]

    # Skip low severity issues (manual review)
    if severity == "Low":
        continue

    column = issue["Column"]
    value = str(dictionary.loc[row, column])

    # -----------------------------
    # Remove digits
    # -----------------------------

    if problem == "Digits Found":

        value = re.sub(r"\d+", "", value)

    # -----------------------------
    # Remove special characters
    # -----------------------------

    elif problem == "Special Character":

        value = re.sub(r"[#@%^&*=+{}<>|~`$]", "", value)

    # -----------------------------
    # Remove duplicate meanings
    # -----------------------------

    elif problem == "Duplicate Meaning":

        meanings = []

        for word in value.split(","):

            word = word.strip()

            if word and word not in meanings:
                meanings.append(word)

        value = ", ".join(meanings)

    # -----------------------------
    # Consecutive commas
    # -----------------------------

    elif problem == "Consecutive Commas":

        value = re.sub(r",+", ",", value)

    # -----------------------------
    # Missing Value
    # -----------------------------

    elif problem == "Missing Value":

        continue

    # =================================================
    # Standard formatting
    # =================================================

    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r",\s*", ", ", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,")

    dictionary.loc[row, column] = value

# =====================================================
# Remove Empty Rows
# =====================================================

dictionary = dictionary[
    dictionary["English"].astype(str).str.strip().ne("")
]

dictionary = dictionary[
    dictionary["Tulu"].astype(str).str.strip().ne("")
]

# =====================================================
# Remove Duplicate Rows
# =====================================================

dictionary.drop_duplicates(inplace=True)
dictionary.reset_index(drop=True, inplace=True)

# =====================================================
# Save
# =====================================================

FINAL_DATA.mkdir(parents=True, exist_ok=True)

dictionary.to_excel(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# Summary
# =====================================================

print("----------------------------------------")
print("Dictionary Fixed Successfully")
print(f"Final Rows : {len(dictionary)}")
print(f"Saved to : {OUTPUT_FILE}")
print("----------------------------------------")