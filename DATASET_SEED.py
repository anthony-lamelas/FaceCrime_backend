#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kagglehub
from pathlib import Path
from PIL import Image
import base64
import torch
import csv
import json
import pandas as pd
import sys

# For Jina CLIP
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModel

def main():
    # ---------------------------------------------------------------------
    # Step 0: Basic settings
    # ---------------------------------------------------------------------
    # Output CSV file with row_id etc.
    final_csv_file = "facecrime_merged_embeddings_fixed_with_id.csv"

    # Path to labels file
    labels_csv_file = "labels_utf8.csv"

    # We'll store embeddings in memory as a list of dicts
    embeddings_data = []

    # ---------------------------------------------------------------------
    # Step 1: Download dataset from KaggleHub
    # ---------------------------------------------------------------------
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("elliotp/idoc-mugshots")
    print("Path to dataset files:", path)
    folder_path = Path(path)

    # ---------------------------------------------------------------------
    # Step 2: Load the embedding model and processor
    # ---------------------------------------------------------------------
    print("Loading processor and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)
    model.eval()

    # ---------------------------------------------------------------------
    # Step 3: Helper functions
    # ---------------------------------------------------------------------
    def load_image_base64(file_path):
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[Base64 Error] {file_path.name}: {e}")
            return None

    def get_embeddings(file_path):
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            with torch.no_grad():
                emb = model.get_image_features(pixel_values=pixel_values)
            emb = emb.squeeze().cpu().numpy().tolist()  # convert to list of floats
            return emb
        except Exception as e:
            print(f"[Embedding Error] {file_path.name}: {e}")
            return None

    # ---------------------------------------------------------------------
    # Step 4: Iterate over images, compute embeddings + base64
    # ---------------------------------------------------------------------
    print("Generating embeddings in memory (no partial CSV) ...")
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue
        # Check if it's a valid image
        try:
            Image.open(file_path).convert("RGB")
        except:
            continue

        image_b64 = load_image_base64(file_path)
        embedding_list = get_embeddings(file_path)
        if image_b64 and embedding_list:
            # We'll store the embedding as a JSON string, so it's easy to keep in CSV
            embeddings_data.append({
                "filename": file_path.name,
                "image_base64": image_b64,
                "embedding": json.dumps(embedding_list)
            })

    print(f"Found {len(embeddings_data)} valid images with embeddings.")

    # Convert to a DataFrame
    embeddings_df = pd.DataFrame(embeddings_data)

    # ---------------------------------------------------------------------
    # Step 5: Merge with labels
    # ---------------------------------------------------------------------
    print(f"Reading labels from: {labels_csv_file}")
    labels_df = pd.read_csv(labels_csv_file)

    print("Merging on filename == ID ...")
    merged_df = pd.merge(
        embeddings_df,
        labels_df,
        how="left",
        left_on="filename",
        right_on="ID"
    )

    # ---------------------------------------------------------------------
    # Step 6: Select + reorder columns
    # ---------------------------------------------------------------------
    # Our final columns:
    # row_id, filename, image_base64, embedding,
    # Sex, Height, Weight, Hair, Eyes, Race, Sex Offender, Offense
    final_columns = [
        "filename",
        "image_base64",
        "embedding",
        "Sex",
        "Height",
        "Weight",
        "Hair",
        "Eyes",
        "Race",
        "Sex Offender",
        "Offense"
    ]

    # Drop the 'ID' column if it exists, we only need 'filename'
    if "ID" in merged_df.columns:
        merged_df.drop(columns=["ID"], inplace=True)

    # Some rows might have no labels => fill them with default empty strings
    for col in ["Sex","Height","Weight","Hair","Eyes","Race","Sex Offender","Offense"]:
        if col not in merged_df.columns:
            merged_df[col] = ""

    # If the columns appear in different names or orders, handle them carefully
    # We'll fill missing columns with an empty string, if needed
    for col in final_columns:
        if col not in merged_df.columns:
            merged_df[col] = ""

    # Reorder
    final_df = merged_df[final_columns].copy()

    # ---------------------------------------------------------------------
    # Step 7: Add row_id
    # ---------------------------------------------------------------------
    final_df.insert(0, "row_id", final_df.index + 1)

    # ---------------------------------------------------------------------
    # Step 8: Write final CSV
    # ---------------------------------------------------------------------
    print(f"Writing final CSV: {final_csv_file}")
    # Ensuring we can handle large fields
    csv.field_size_limit(sys.maxsize)

    final_df.to_csv(final_csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✅ Done! Created {final_csv_file} with row_id and all columns.")


if __name__ == "__main__":
    main()
