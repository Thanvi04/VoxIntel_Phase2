import pandas as pd
import re

from config import CLEANED_DATA, OUTPUT_DATA

# =====================================================
# File Paths
# =====================================================

INPUT_FILE = CLEANED_DATA / "clean_dictionary.xlsx"
OUTPUT_FILE = OUTPUT_DATA / "validation_report.xlsx"

# =====================================================
# Load Dataset
# =====================================================

print("Loading cleaned dictionary...")

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()

print(f"Rows Loaded : {len(df)}")

# =====================================================
# Validation Report
# =====================================================

validation = []

# =====================================================
# Validation Loop
# =====================================================

for index, row in df.iterrows():

    english = str(row["English"]).strip()
    tulu = str(row["Tulu"]).strip()

    excel_row = index + 2

    # -----------------------------
    # Missing English
    # -----------------------------

    if english == "" or english.lower() == "nan":

        validation.append([
            excel_row,
            "English",
            english,
            "Missing Value",
            "Fill the missing English word",
            "High"
        ])

    # -----------------------------
    # Missing Tulu
    # -----------------------------

    if tulu == "" or tulu.lower() == "nan":

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Missing Value",
            "Fill the missing Tulu meaning",
            "High"
        ])

    # -----------------------------
    # English inside Tulu
    # -----------------------------

    if re.search(r"[A-Za-z]", tulu):

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "English Found in Tulu",
            "Review manually",
            "Low"
        ])

    # -----------------------------
    # Digits in English
    # -----------------------------

    if re.search(r"\d", english):

        validation.append([
            excel_row,
            "English",
            english,
            "Digits Found",
            "Remove digits",
            "Medium"
        ])

    # -----------------------------
    # Digits in Tulu
    # -----------------------------

    if re.search(r"\d", tulu):

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Digits Found",
            "Remove digits",
            "Medium"
        ])

    # -----------------------------
    # Special Characters
    # -----------------------------

    if re.search(r"[#@%^&*=+{}<>|~`$]", tulu):

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Special Character",
            "Remove special character",
            "Medium"
        ])

    # -----------------------------
    # Duplicate Meanings
    # -----------------------------

    meanings = [x.strip() for x in tulu.split(",") if x.strip()]

    if len(meanings) != len(set(meanings)):

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Duplicate Meaning",
            "Keep one occurrence",
            "Medium"
        ])

    # -----------------------------
    # Consecutive Commas
    # -----------------------------

    if ",," in tulu:

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Consecutive Commas",
            "Replace with single comma",
            "Medium"
        ])

    # -----------------------------
    # Multiple Spaces
    # -----------------------------

    if re.search(r"\s{2,}", tulu):

        validation.append([
            excel_row,
            "Tulu",
            tulu,
            "Multiple Spaces",
            "Replace with single space",
            "Low"
        ])

# =====================================================
# Save Report
# =====================================================

report = pd.DataFrame(
    validation,
    columns=[
        "Row",
        "Column",
        "Original Value",
        "Issue",
        "Suggested Fix",
        "Severity"
    ]
)

OUTPUT_DATA.mkdir(parents=True, exist_ok=True)

report.to_excel(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# Summary
# =====================================================

print("----------------------------------------")
print("Validation Completed Successfully")
print(f"Issues Found : {len(report)}")
print(f"Report Saved : {OUTPUT_FILE}")

print("\nIssue Summary")
print(report["Issue"].value_counts())

print("\nSeverity Summary")
print(report["Severity"].value_counts())