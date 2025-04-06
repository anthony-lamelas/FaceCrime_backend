# api/embedding.py

import base64
import requests
from PIL import Image
from io import BytesIO
from typing import Union, List
import os 
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

JINA_API_KEY = os.environ.get('JINA_API_KEY')
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY not found in environment variables.")

# Define headers with API authorization token                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
API_URL = 'https://api.jina.ai/v1/embeddings'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {JINA_API_KEY}'
}

# Define standard headers for image requests
IMAGE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Resize image to 224x224 using Lanczos filter
def resize_image(image_data: bytes) -> str:
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to handle a single input or batch query for embedding
def process_embedding(input_data: Union[str, dict, List[Union[str, dict]]]):
    # Prepare input format for Jina API
    def prepare_input(data):
        if isinstance(data, str):
            if data.startswith('http'):
                # Image URL: Fetch and resize with error handling
                logger.info("Processing as image URL")
                try:
                    image_response = requests.get(data, headers=IMAGE_HEADERS, timeout=5)
                    if image_response.status_code == 200:
                        resized_image_base64 = resize_image(image_response.content)
                        return {"image": resized_image_base64}
                    else:
                        logger.error(f"Failed to load image from URL: {data} - Status Code: {image_response.status_code}")
                        return None  # Skip this image if it can't be loaded
                except requests.exceptions.RequestException as e:
                    logger.error(f"Exception while loading image: {e}")
                    return None  # Skip this image if an exception occurs
            elif data.startswith('/9j/') or data.startswith('R0lGOD') or data.startswith("data:image"):
                # Base64 image: Resize
                logger.info("Processing as base64 image")
                try:
                    # Handle both formats: with data:image prefix or just the base64 string
                    if data.startswith("data:image"):
                        # Split by the base64 indicator
                        image_data = base64.b64decode(data.split(",")[1])
                    else:
                        image_data = base64.b64decode(data)
                    resized_image_base64 = resize_image(image_data)
                    return {"image": resized_image_base64}
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    return None  # Skip this image if decoding fails
            else:
                # Text input
                logger.info("Processing as text")
                return {"text": data}
        elif isinstance(data, dict):
            # If it's already a dict with 'image_or_text' key
            if 'image_or_text' in data:
                return prepare_input(data['image_or_text'])
            return {key: prepare_input(value) for key, value in data.items()}
        else:
            logger.warning("Unsupported input type")
            return None  # Skip unsupported input types

    # If input_data is a single item, wrap it in a list for batch processing
    if not isinstance(input_data, list):
        input_data = [input_data]

    # Format input data for the API, skipping any None results from failed images
    inputs = [prepared for prepared in (prepare_input(item) for item in input_data) if prepared]
    logger.info(f"Formatted {len(inputs)} inputs for Jina API")

    if not inputs:
        logger.error("No valid inputs could be prepared for the embedding API")
        return None

    # Data payload for API request
    data = {
        "model": "jina-clip-v1",
        "normalized": True,
        "embedding_type": "float",
        "input": inputs
    }

    # Send request to Jina's Clip API
    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        logger.info(f"Jina API response status code: {response.status_code}")
        
        if response.status_code == 200:
            # Extract embeddings from the nested structure
            response_data = response.json().get("data", [])
            embeddings = [entry.get("embedding") for entry in response_data]
            return embeddings[0] if len(embeddings) == 1 else embeddings  # Return single embedding if only one input
        else:
            logger.error(f"Failed to get embeddings: {response.status_code} - {response.text}")
            raise ValueError(f"Failed to get embeddings: {response.status_code} - {response.text}")
    except Exception as e:
        logger.exception(f"Exception during Jina API call: {e}")
        raise
