import os
import logging
import datetime
from fastapi import HTTPException
from motor import motor_asyncio

logger = logging.getLogger(__name__)

# The default is local Mongo (community), named "facecrime"
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongo:27017/facecrime")
logger.info(f"Using MONGODB_URI={MONGODB_URI}")

try:
    client = motor_asyncio.AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.get_database()  # if using facecrime in the URI, that’s your default
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise HTTPException(status_code=500, detail="Database connection error")

async def ensure_vector_index():
    """
    Attempts to create a HNSW vector index for Atlas or Enterprise 7.0+.
    If running local/community, it fails and logs a warning—then we rely on fallback.
    """
    try:
        collection = db.images
        existing_indexes = await collection.index_information()
        if "embedding_vector_index" not in existing_indexes:
            logger.info("Attempting to create vector search index for embeddings...")
            await collection.create_index(
                [("embedding", "vectorHNSW")],
                name="embedding_vector_index",
                vectorDimension=512,
                vectorDistanceMetric="cosine"
            )
            logger.info("HNSW vector index created successfully")
    except Exception as e:
        logger.warning(
            "Vector index creation failed (likely running community Mongo). "
            "Falling back to basic search.\n"
            f"Original error: {str(e)}"
        )
        # Do NOT re-raise; just skip so the rest can run

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
    with the vector 'embedding'.
    """
    try:
        # Attempt vector index creation if missing
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
    Attempts to do a $vectorSearch. If it fails (community),
    we fallback to a basic find() query.
    """
    try:
        await ensure_vector_index()
        collection = db.images

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,  # or 150000, your call
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
        logger.warning(f"Vector search failed, falling back to basic find query: {str(e)}")
        try:
            documents = await collection.find().limit(limit*10).to_list(length=limit*10)
            if not documents:
                return []
            # Just return some documents without real similarity ranking
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
