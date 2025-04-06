# services/database.py

import os
import logging
import datetime
from fastapi import HTTPException
from motor import motor_asyncio

# Initialize the logger
logger = logging.getLogger(__name__)

# MongoDB Atlas connection string - should be in environment variables
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    logger.warning("MONGODB_URI not found in environment variables. Using default connection string.")
    MONGODB_URI = "mongodb://mongo:27017/business"
    # why /business?

# Create a cached client and database instance
try:
    client = motor_asyncio.AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.business  # Use the 'business' database
    logger.info("Connected to MongoDB Atlas successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
    raise HTTPException(status_code=500, detail="Database connection error")


# Create vector index if it doesn't exist
async def ensure_vector_index():
    """Ensure the vector search index exists for the images collection"""
    try:
        collection = db.images
        # Check if the index already exists
        existing_indexes = await collection.index_information()
        if "embedding_vector_index" not in existing_indexes:
            logger.info("Creating vector search index for embeddings...")
            # Create a 2dsphere index for the embedding field
            await collection.create_index(
                [("embedding", "vectorHNSW")],
                name="embedding_vector_index",
                vectorDimension=512,  # Jina embeddings are typically 512 dimensions, adjust if different
                vectorDistanceMetric="cosine"
            )
            logger.info("Vector search index created successfully")
    except Exception as e:
        logger.error(f"Failed to create vector index: {str(e)}")
        # Don't raise an exception here, as this is initialization
        # The search will still work without the index, just slower


async def insert_image(image_base64: str, embedding: list):
    """Insert a new image with its embedding into the database"""
    try:
        # Ensure vector index exists
        await ensure_vector_index()
        
        collection = db.images
        document = {
            "image_base64": image_base64,
            "embedding": embedding,
            "created_at": datetime.datetime.now()
        }
        result = await collection.insert_one(document)
        logger.info(f"Inserted image with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to insert image into database: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")


async def find_similar_image(embedding: list, limit: int = 1):
    """Find the most similar image(s) based on embedding similarity using MongoDB's vector search"""
    try:
        # Ensure vector index exists
        await ensure_vector_index()
        
        collection = db.images
        
        # Use MongoDB's $vectorSearch aggregation stage for similarity search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,  # Adjust based on your collection size
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "image_base64": 1,
                    "similarity_score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]
        
        # Execute the aggregation pipeline
        cursor = collection.aggregate(pipeline)
        documents = await cursor.to_list(length=limit)
        
        if not documents:
            logger.warning("No similar images found in the database")
            return []
        
        # Format the results
        results = []
        for doc in documents:
            results.append({
                "id": str(doc.get("_id")),
                "image_base64": doc.get("image_base64"),
                "similarity_score": float(doc.get("similarity_score", 0))
            })
        
        return results
    except Exception as e:
        logger.error(f"Failed to find similar images: {str(e)}")
        # If vectorSearch fails (e.g., if index doesn't exist), fallback to simpler query
        try:
            logger.warning("Vector search failed, falling back to basic find query")
            # Get all documents (not efficient, but a fallback)
            documents = await collection.find().limit(limit * 10).to_list(length=limit * 10)
            
            if not documents:
                return []
            
            # Just return some documents without similarity score
            results = []
            for doc in documents[:limit]:
                results.append({
                    "id": str(doc.get("_id")),
                    "image_base64": doc.get("image_base64"),
                    "similarity_score": 0.0  # We don't have a real score
                })
            
            return results
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail="Database search operation failed")
