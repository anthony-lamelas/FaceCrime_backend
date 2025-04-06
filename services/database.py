import os
import logging
import datetime
from fastapi import HTTPException
from motor import motor_asyncio

# For Milvus integration
from pymilvus import (
    Collection, 
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    connections, 
    utility
)

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongo:27017/business")
logger.info(f"Using MONGODB_URI={MONGODB_URI}")

try:
    client = motor_asyncio.AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.business  # 'business' database
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise HTTPException(status_code=500, detail="Database connection error")

# Milvus connection parameters
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_URI = f"{MILVUS_HOST}:{MILVUS_PORT}"
MILVUS_COLLECTION_NAME = "images_collection"

# Connect to Milvus
try:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    # Create collection in Milvus if it doesn't exist
    if MILVUS_COLLECTION_NAME not in utility.list_collections():
        fields = [
            FieldSchema(name="mongo_id", dtype=DataType.VARCHAR, max_length=24, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
        ]
        schema = CollectionSchema(fields, description="Images collection for vector search")
        Collection(name=MILVUS_COLLECTION_NAME, schema=schema)
        logger.info("Milvus collection created")
    else:
        logger.info("Milvus collection already exists")
except Exception as e:
    logger.error(f"Failed to connect or initialize Milvus: {str(e)}")
    raise HTTPException(status_code=500, detail="Milvus connection error")


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
    Insert a new image + metadata into MongoDB (for metadata) and Milvus (for vector search).
    """
    try:
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
        inserted_id = str(result.inserted_id)
        logger.info(f"Inserted image + metadata with ID: {inserted_id}")
    except Exception as e:
        logger.error(f"Failed to insert image metadata into MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="MongoDB insertion failed")

    # Insert vector into Milvus
    try:
        milvus_collection = Collection(name=MILVUS_COLLECTION_NAME)
        # Insert expects a list of lists for each field.
        entities = [
            [inserted_id],         # mongo_id field (as string)
            [embedding]            # embedding field (list of float)
        ]
        milvus_collection.insert(entities)
        milvus_collection.flush()
        logger.info(f"Inserted vector for image ID: {inserted_id} into Milvus")
    except Exception as e2:
        logger.error(f"Failed to insert vector into Milvus: {str(e2)}")
        # Optionally, decide whether to fail entirely or continue.
    return inserted_id


async def find_similar_image(embedding: list, limit: int = 1):
    """
    Find the most similar images by performing a vector search via Milvus.
    Retrieves the corresponding metadata from MongoDB.
    """
    try:
        # Load collection into memory first (required for search)
        milvus_collection = Collection(name=MILVUS_COLLECTION_NAME)
        milvus_collection.load()
        
        # Search parameters for HNSW index
        search_params = {
            "metric_type": "COSINE", 
            "params": {"ef": 128}
        }
        
        # Perform the search
        search_results = milvus_collection.search(
            data=[embedding],  # Query vector
            anns_field="embedding",  # Vector field to search
            param=search_params,
            limit=limit,  # Number of results to return
            output_fields=["mongo_id"]  # Fields to return
        )
        
        # Process Milvus search results - using try/except to safely handle any type issues
        try:
            # Assuming search_results is already the properly structured results
            # Since Milvus API might vary across versions, we'll use a more robust approach
            
            # First, verify we have results
            if not search_results:
                logger.warning("No results returned from Milvus")
                return []
                
            # Extract MongoDB IDs from the search results
            mongo_ids = []
            
            # Try to process as a standard result struct
            # Access first query result (we only submitted one query vector)
            query_hits = None
            
            # Different ways to access results depending on Milvus version
            if isinstance(search_results, list) and len(search_results) > 0:
                # Direct list access for newer versions
                query_hits = search_results[0]
            
            # Check if we have any results
            if not query_hits:
                logger.warning("No hits found in Milvus search results")
                return []
                
            # Debug information
            logger.debug(f"Found hits in Milvus: {type(query_hits)}")
            
            # Extract MongoDB IDs from each hit
            for hit in query_hits:
                try:
                    # Access entity data if available
                    if hasattr(hit, 'entity') and hasattr(hit.entity, 'get'):
                        mongo_id = hit.entity.get('mongo_id')
                        if mongo_id:
                            mongo_ids.append(mongo_id)
                except Exception as hit_error:
                    logger.warning(f"Error processing hit: {str(hit_error)}")
                    continue
                    
            if not mongo_ids:
                logger.warning("No valid MongoDB IDs found in search results")
                return []
                
        except Exception as process_error:
            logger.error(f"Error processing Milvus search results: {str(process_error)}")
            return []
        
        # We've already collected mongo_ids at this point
        # No need for additional extraction
        
        # Fetch metadata from MongoDB using the retrieved IDs
        collection = db.images
        # Convert string IDs to ObjectId if necessary; here we assume string IDs are used
        cursor = collection.find({"_id": {"$in": mongo_ids}})
        documents = await cursor.to_list(length=limit)
        results = []
        for doc in documents:
            results.append({
                "id": str(doc.get("_id")),
                "image_base64": doc.get("image_base64"),
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
        logger.error(f"Failed to find similar images via Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail="Vector search operation failed")
