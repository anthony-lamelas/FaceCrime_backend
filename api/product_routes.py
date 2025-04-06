import logging
from fastapi import APIRouter, HTTPException, Request, Response
from api.embedding import process_embedding
from services.database import find_similar_image, insert_image
from typing import Optional
import base64

# Initialize the logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/submission")
async def submission(request: Request):
    """Process an incoming image, get its embedding, and find the most similar face in the database"""
    try:
        # Receive JSON data from frontend (React)
        data = await request.json()
        
        # Validate the input data
        if 'image_or_text' not in data:
            logger.error("Missing 'image_or_text' field in request")
            raise HTTPException(status_code=400, detail="Missing 'image_or_text' field in request")
        
        # Log that we received an image (but don't log the entire base64 string)
        logger.info("Received image submission request")
        
        # Process the embedding through Jina API
        user_input_embedding = process_embedding(data['image_or_text'])
        
        if not user_input_embedding:
            logger.error("Failed to generate embedding for the input image")
            raise HTTPException(status_code=400, detail="Failed to process image")
            
        logger.info("Successfully generated embedding from input image")
        
        # Search MongoDB for similar faces using the embedding
        similar_images = await find_similar_image(user_input_embedding)
        
        if not similar_images:
            logger.warning("No similar images found in the database")
            return {"results": []}
        
        # Return the most similar image
        most_similar = similar_images[0]
        logger.info(f"Found similar image with similarity score: {most_similar['similarity_score']}")
        
        # Return the matching image to the frontend
        return {
            "results": [{
                "id": most_similar["id"],
                "image": most_similar["image_base64"],
                "similarity_score": most_similar["similarity_score"]
            }]
        }
    except Exception as e:
        # Log the error with traceback for debugging
        logger.exception(f"An error occurred while processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/add-image")
async def add_image(request: Request):
    """Add a new image to the database with its embedding"""
    try:
        # Receive JSON data
        data = await request.json()
        
        # Validate input
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
            
        # Generate embedding for the image
        embedding = process_embedding(data['image'])
        
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding for the image")
        
        # Insert the image and its embedding into the database
        result_id = await insert_image(data['image'], embedding)
        
        return {"id": result_id, "message": "Image added successfully"}
    except Exception as e:
        logger.exception(f"Error adding image to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add image: {str(e)}")