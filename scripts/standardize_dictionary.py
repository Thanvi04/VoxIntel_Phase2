import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned" / "clean_dictionary.xlsx"

OUTPUT_FILE = BASE_DIR / "data" / "final" / "final_dictionary.xlsx"

df = pd.read_excel(INPUT_FILE)

df.columns = df.columns.str.strip()

# ----------------------------------------
# Standardization
# ----------------------------------------

def clean_text(text):

    text = str(text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # remove spaces before commas
    text = re.sub(r"\s+,", ",", text)

    # single space after comma
    text = re.sub(r",\s*", ", ", text)

    return text.strip()

df["English"] = df["English"].apply(clean_text)
df["Tulu"] = df["Tulu"].apply(clean_text)

# Remove duplicate rows again

df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

OUTPUT_FILE.parent.mkdir(exist_ok=True)

df.to_excel(OUTPUT_FILE, index=False)

print("--------------------------------")
print("Standardization Completed")
print(f"Final Rows : {len(df)}")
print(f"Saved : {OUTPUT_FILE}")