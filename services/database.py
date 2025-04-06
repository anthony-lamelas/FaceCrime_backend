import os
import logging
import datetime
from fastapi import HTTPException
from motor import motor_asyncio

# Initialize the logger
logger = logging.getLogger(__name__)

# MongoDB Atlas or local connection string (from environment)
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongo:27017/business")
logger.info(f"Using MONGODB_URI={MONGODB_URI}")

# Create a cached client and database instance
try:
    client = motor_asyncio.AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.business  # 'business' database
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise HTTPException(status_code=500, detail="Database connection error")


async def ensure_vector_index():
    """
    Ensure that we have a HNSW vector index on the 'embedding' field of the 'images' collection.
    """
    try:
        collection = db.images
        existing_indexes = await collection.index_information()
        if "embedding_vector_index" not in existing_indexes:
            logger.info("Creating vector search index for embeddings...")
            # Create the HNSW vector index
            await collection.create_index(
                [("embedding", "vectorHNSW")],
                name="embedding_vector_index",
                # Make sure this dimension matches your actual embedding size from CLIP
                vectorDimension=512,  
                vectorDistanceMetric="cosine"
            )
            logger.info("HNSW vector index created successfully")
    except Exception as e:
        logger.error(f"Failed to create vector index: {str(e)}")
        # We'll not raise, so the system can still run without the index (though searches might fail or fallback).


async def insert_image_and_metadata(
    image_base64: str,
    embedding: list,
    sex: str,
    height: str,
    weight: str,
    hairColor: str,
    eyeColor: str,
    race: str,
    sexOffender: bool,
    offense: str
):
    """
    Insert a new image + metadata into the DB,
    with the vector 'embedding' for vector search,
    plus all the other fields for retrieval.
    """
    try:
        # Ensure vector index
        await ensure_vector_index()
        
        collection = db.images

        document = {
            "image_base64": image_base64,
            "embedding": embedding,  # 512-d float array
            "sex": sex,
            "height": height,
            "weight": weight,
            "hairColor": hairColor,
            "eyeColor": eyeColor,
            "race": race,
            "sexOffender": sexOffender,
            "offense": offense,
            "created_at": datetime.datetime.utcnow()
        }

        result = await collection.insert_one(document)
        logger.info(f"Inserted image + metadata with ID: {result.inserted_id}")
        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"Failed to insert image metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")


async def find_similar_image(embedding: list, limit: int = 1):
    """
    Find the most similar image(s) by performing a vector search.
    Return all relevant metadata, so the route can shape the response.
    """
    try:
        await ensure_vector_index()
        collection = db.images

        # Use MongoDB $vectorSearch. 
        # Adjust numCandidates if your collection is large or small.
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 150000,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "image_base64": 1,
                    "sex": 1,
                    "height": 1,
                    "weight": 1,
                    "hairColor": 1,
                    "eyeColor": 1,
                    "race": 1,
                    "sexOffender": 1,
                    "offense": 1,
                    "similarity_score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        documents = await cursor.to_list(length=limit)

        if not documents:
            logger.warning("No similar images found in the database")
            return []

        results = []
        for doc in documents:
            results.append({
                "id": str(doc.get("_id")),
                "image_base64": doc.get("image_base64"),
                "similarity_score": float(doc.get("similarity_score", 0)),
                "sex": doc.get("sex"),
                "height": doc.get("height"),
                "weight": doc.get("weight"),
                "hairColor": doc.get("hairColor"),
                "eyeColor": doc.get("eyeColor"),
                "race": doc.get("race"),
                "sexOffender": doc.get("sexOffender"),
                "offense": doc.get("offense")
            })
        return results

    except Exception as e:
        logger.error(f"Failed to find similar images: {str(e)}")
        # fallback if vectorSearch fails (e.g. no index)
        try:
            logger.warning("Vector search failed, falling back to basic find query")
            documents = await collection.find().limit(limit*10).to_list(length=limit*10)
            if not documents:
                return []
            results = []
            for doc in documents[:limit]:
                results.append({
                    "id": str(doc.get("_id")),
                    "image_base64": doc.get("image_base64"),
                    "similarity_score": 0.0,
                    "sex": doc.get("sex"),
                    "height": doc.get("height"),
                    "weight": doc.get("weight"),
                    "hairColor": doc.get("hairColor"),
                    "eyeColor": doc.get("eyeColor"),
                    "race": doc.get("race"),
                    "sexOffender": doc.get("sexOffender"),
                    "offense": doc.get("offense")
                })
            return results
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail="Database search operation failed")
