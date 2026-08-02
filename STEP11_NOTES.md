# Step 11 — Final Dataset Quality Filtering

## What this step does
Takes `output/final_training_dataset_cleaned.jsonl` (output of step 6/7,
60,778 records, 4,002 flagged as "suspicious" by `validate_training_dataset.py`)
and actually fixes / removes the flagged issues, producing a dataset that is
ready for step 12 (revalidate + recreate splits).

## Script
`scripts/filter_final_dataset.py`

Run it with:
```
python scripts/filter_final_dataset.py
```

## What it fixes automatically
- Collapses repeated / mixed punctuation runs (`..`, `,,,`, `.,.`) into a
  single mark.
- Removes leftover dictionary abbreviations (`adj.`, `n.`, `s.`, etc.).
- Strips out stray Latin/English tokens embedded in Tulu/Kannada responses
  (OCR / scraping noise such as `AR`, `half-grown-`, `WRIFE`).
- Re-normalizes punctuation spacing and whitespace.

## What it removes (and logs for manual review)
- Records whose response becomes empty or shorter than 2 characters after
  cleaning (e.g. response was only `"y"`, `"_"`, or a single stray letter) —
  these cannot be salvaged automatically.
- Any new exact duplicates created as a side effect of cleaning.

Removed records are **never silently discarded** — they're saved to
`output/removed_records_review.jsonl` so they can be manually checked later.

## Results (this run)
| Metric | Value |
|---|---|
| Records loaded | 60,778 |
| Records kept (filtered dataset) | 60,074 |
| Records removed (empty/too short) | 704 |
| Records removed (new duplicates) | 0 |
| Records auto-fixed | 4,446 |
| Remaining English/Latin chars in responses | 0 |
| Remaining repeated punctuation | 0 |

## Output files
- `output/final_training_dataset_filtered.jsonl` — use this as the input to
  **step 12** (revalidate + recreate splits), instead of
  `final_training_dataset_cleaned.jsonl`.
- `output/removed_records_review.jsonl` — 704 records that were dropped;
  worth a quick manual skim in case anything meaningful got cut.
- `output/final_filtering_report.txt` — summary report (same info as above).

## Next step (12)
Point `split_training_dataset.py` and `validate_training_dataset.py` at
`final_training_dataset_filtered.jsonl` instead of the old cleaned file, and
re-generate `train.jsonl` / `validation.jsonl` / `test.jsonl`.
