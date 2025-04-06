import logging
from fastapi import APIRouter, HTTPException, Request
from api.embedding import process_embedding
from services.database import find_similar_image, insert_image_and_metadata
from typing import Optional

# Initialize the logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/submission")
async def submission(request: Request):
    """
    Process incoming image, get its embedding, 
    find the most similar face in the DB, 
    and return all attributes in the JSON format the frontend expects.
    """
    try:
        data = await request.json()
        
        if 'image_or_text' not in data:
            logger.error("Missing 'image_or_text' field in request")
            raise HTTPException(status_code=400, detail="Missing 'image_or_text' field in request")
        
        # Generate embedding from the image
        user_input_embedding = process_embedding(data['image_or_text'])
        if not user_input_embedding:
            logger.error("Failed to generate embedding for the input image")
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Find the most similar criminal
        similar_images = find_similar_image(user_input_embedding, limit=1)
        if not similar_images:
            logger.warning("No similar images found in the database")
            # Return an empty result or some default
            return {"results": []}
        
        most_similar = similar_images[0]
        logger.info(f"Found similar image with similarity score: {most_similar['similarity_score']}")

        # Convert similarity_score -> matchPercent (0-1 scale or 0-100). 
        # For example, if your vectorDistanceMetric="cosine", 
        # similarity_score might be the dot product or something. 
        # You can interpret it or scale it as you prefer:
        matchPercent = round(most_similar["similarity_score"], 3)

        # Return the JSON in the exact shape your frontend wants
        # Example:
        response_json = {
            "image": most_similar["image_base64"],
            "offense": most_similar["offense"],
            "height": most_similar["height"],
            "weight": most_similar["weight"],
            "hairColor": most_similar["hairColor"],
            "eyeColor": most_similar["eyeColor"],
            "race": most_similar["race"],
            "sexOffender": most_similar["sexOffender"],
            "matchPercent": matchPercent
        }

        return response_json

    except Exception as e:
        logger.exception(f"An error occurred while processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/add-image")
async def add_image(request: Request):
    """
    Endpoint to insert a new image + metadata directly from the frontend.
    This is optional if you prefer the offline 'seed_data.py' approach.
    """
    try:
        data = await request.json()

        required_fields = ["image", "embedding", "offense", "height", 
                           "weight", "hairColor", "eyeColor", "race", 
                           "sexOffender"]
        
        # Basic validation
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing '{field}' in request payload"
                )
        
        # Insert into DB
        inserted_id = insert_image_and_metadata(
            image_base64=data["image"],
            embedding=data["embedding"],
            sex=data.get("sex", "Unknown"),  # optional
            height=data["height"],
            weight=data["weight"],
            hairColor=data["hairColor"],
            eyeColor=data["eyeColor"],
            race=data["race"],
            sexOffender=bool(data["sexOffender"]),
            offense=data["offense"],
            filename=data["filename"]
        )
        
        return {"id": inserted_id, "message": "Image + metadata added successfully"}

    except Exception as e:
        logger.exception(f"Error adding image to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add image: {str(e)}")
