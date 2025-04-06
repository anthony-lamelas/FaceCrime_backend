import os
import logging
import datetime
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Build the Postgres connection string from environment
MANUFACTURER_DB_HOST = os.environ.get("MANUFACTURER_DB_HOST", "localhost")
MANUFACTURER_DB_DB = os.environ.get("MANUFACTURER_DB_DB", "facecrime")
MANUFACTURER_DB_USER = os.environ.get("MANUFACTURER_DB_USER", "facecrimeuser")
MANUFACTURER_DB_PASSWORD = os.environ.get("MANUFACTURER_DB_PASSWORD", "facecrimepass")

conn_str = (
    f"host={MANUFACTURER_DB_HOST} dbname={MANUFACTURER_DB_DB} user={MANUFACTURER_DB_USER} password={MANUFACTURER_DB_PASSWORD}"
)

def get_connection():
    return psycopg2.connect(conn_str)

def insert_image_and_metadata(
    filename: str,
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
    Insert a row into the 'images' table with a 768-d vector, storing
    everything in Postgres + pgvector. 'filename' is the PK.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # We'll store the embedding as an ARRAY -> cast to vector in SQL
                # or let psycopg2 handle it if we have correct extension
                sql = """
                INSERT INTO images
                    (filename, image_base64, embedding, sex, height, weight,
                     hairColor, eyeColor, race, sexOffender, offense, created_at)
                VALUES
                    (%s, %s, %s::vector(768), %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (filename) DO NOTHING
                """
                cur.execute(sql, (
                    filename,
                    image_base64,
                    embedding,  # we will pass a python list, must convert to string or do psycopg2 adaptation
                    sex,
                    height,
                    weight,
                    hairColor,
                    eyeColor,
                    race,
                    sexOffender,
                    offense
                ))
        logger.info(f"Inserted data for filename='{filename}'")
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")

def find_similar_image(embedding: list, limit: int = 1):
    """
    Perform a similarity search using pgvector operator (<->).
    Lower distance = more similar if using L2 or IP, 
    or you can store normalized embeddings and do l2.
    
    For example, if you used L2 distance:
      ORDER BY embedding <-> %s::vector(768)
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Example: L2 distance
                sql = f"""
                SELECT
                  filename,
                  image_base64,
                  sex,
                  height,
                  weight,
                  hairColor,
                  eyeColor,
                  race,
                  sexOffender,
                  offense,
                  (embedding <-> %s::vector(768)) AS matchPercent
                FROM images
                ORDER BY embedding <-> %s::vector(768)
                LIMIT {limit};
                """
                cur.execute(sql, (embedding, embedding))
                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "filename": row["filename"],
                        "image_base64": row["image_base64"],
                        "sex": row["sex"],
                        "height": row["height"],
                        "weight": row["weight"],
                        "hairColor": row["haircolor"],
                        "eyeColor": row["eyecolor"],
                        "race": row["race"],
                        "sexOffender": row["sexoffender"],
                        "offense": row["offense"],
                        "matchPercent": float(row["matchPercent"]),
                    })
                return results
    except Exception as e:
        logger.error(f"Failed to query similar images: {e}")
        return []
