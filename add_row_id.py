#!/usr/bin/env python3
import sys
import csv

INPUT_CSV = "/tmp/facecrime_merged_embeddings_fixed.csv"
OUTPUT_CSV = "/tmp/facecrime_merged_embeddings_fixed_with_id.csv"

csv.field_size_limit(sys.maxsize)

def main():
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f_in, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        # Prepend a new column name 'row_id' to the existing fieldnames
        fieldnames = ["row_id"] + reader.fieldnames
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        row_id_counter = 1
        for row in reader:
            # Insert the new 'row_id' key-value before writing
            row["row_id"] = row_id_counter
            writer.writerow(row)
            row_id_counter += 1

    print(f"✅ Done. New CSV with 'row_id' created at: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

