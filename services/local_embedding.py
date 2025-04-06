import os
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io
import logging
import torch
from transformers.models.auto.modeling_auto import AutoModel

# Initialize the model and move it to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)


# Function to generate embeddings
def generate_embedding(img):
    try:
        # Encode the image
        image_embedding = model.encode_image([img])  # returns a NumPy array
        logging.info("Successfully generated embedding from input image")
        # Use the embeddings directly
        image_vector = image_embedding[0]

        return image_vector
    except Exception as e:
        logging.error(f"Error generating embeddings for image: {str(e)}")
        return None