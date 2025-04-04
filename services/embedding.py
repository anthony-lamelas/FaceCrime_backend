# services/embedding.py

import base64
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
from typing import Union, List, Dict
import logging
import os

API_URL = 'https://api.jina.ai/v1/embeddings'

# Get the Jina API key from environment variables
JINA_API_KEY = os.environ.get('JINA_API_KEY')
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY not found in environment variables.")

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {JINA_API_KEY}'
}

IMAGE_HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://www.bestbuy.com',
}

logger = logging.getLogger(__name__)

async def resize_image(image_data: bytes) -> str:
    """
    Resize image to 224x224 using PIL and return base64 string.
    """
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224), Image.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def process_embedding(input_data: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
                            model: str,
                            source: str = 'api') -> List[List[float]]:
    """
    Process embeddings for input data using Jina's API.

    Args:
        input_data (str, dict, or list): Input data (text or image data).
        model (str): Model to use ('jina-clip-v1' for images and text, 'jina-embeddings-v3' for text).
        source (str): 'user' or 'api', indicating the source of the image.

    Returns:
        List[List[float]]: List of embeddings.
    """
    async def prepare_input(data):
        if model == 'jina-clip-v1':
            # Image or text embeddings
            if isinstance(data, dict):
                # Image from API: {'image': 'image_url'}
                image_url = data.get('image')
                if not image_url:
                    logger.error("Image URL is None")
                    return None
                if image_url.startswith('http'):
                    # Image URL: Fetch, resize, and prepare
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url, headers=IMAGE_HEADERS, timeout=10) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    resized_image_base64 = await resize_image(image_data)
                                    return {"image": resized_image_base64}
                                else:
                                    logger.error(f"Failed to fetch image from URL: {image_url}, Status: {resp.status}")
                                    return None
                    except Exception as e:
                        logger.error(f"Exception while fetching image from URL: {image_url}, Error: {e}")
                        return None
                else:
                    # Assume it's already base64 string
                    return {"image": data.get('image')}
            elif isinstance(data, str):
                if data.startswith('data:image'):
                    # User image: base64 string
                    _, base64_data = data.split(',', 1)
                    return {"image": base64_data}
                else:
                    # Assume it's text
                    return {"text": data}
            else:
                logger.error("Unsupported input type for jina-clip-v1 model")
                return None
        elif model == 'jina-embeddings-v3':
            # Text embeddings
            if isinstance(data, str):
                return data
            else:
                logger.error("Unsupported input type for jina-embeddings-v3 model")
                return None
        else:
            raise ValueError(f"Unsupported model: {model}")

    # Prepare inputs
    if not isinstance(input_data, list):
        input_data = [input_data]

    prepared_inputs = []
    for item in input_data:
        prepared = await prepare_input(item)
        if prepared:
            prepared_inputs.append(prepared)
        else:
            logger.error(f"Failed to prepare input: {item}")

    # Check if prepared_inputs is not empty
    if not prepared_inputs:
        logger.error("No valid inputs were prepared for embedding.")
        return None

    # Depending on model, prepare the JSON differently
    if model == 'jina-clip-v1':
        # Image or text embeddings
        json_data = {
            "model": model,
            "normalized": True,
            "embedding_type": "float",
            "input": prepared_inputs  # list of {"image": base64string} or {"text": "text"}
        }
    elif model == 'jina-embeddings-v3':
        # Text embeddings
        json_data = {
            "model": model,
            "task": "text-matching",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": prepared_inputs  # list of strings
        }
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Send request to Jina API
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=HEADERS, json=json_data) as response:
            if response.status == 200:
                response_data = await response.json()
                embeddings = [entry.get("embedding") for entry in response_data.get("data", [])]
                return embeddings
            else:
                text = await response.text()
                logger.error(f"Failed to get embeddings: {response.status} - {text}")
                return None
