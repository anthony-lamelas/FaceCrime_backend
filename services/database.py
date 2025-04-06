import os
import logging
import datetime
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Build Postgres connection string from environment
DB_HOST = os.environ.get("MANUFACTURER_DB_HOST", "localhost")
DB_PORT = os.environ.get("MANUFACTURER_DB_PORT", "5432")
DB_NAME = os.environ.get("MANUFACTURER_DB_DB", "facecrime")
DB_USER = os.environ.get("MANUFACTURER_DB_USER", "facecrimeuser")
DB_PASSWORD = os.environ.get("MANUFACTURER_DB_PASSWORD", "facecrimepass")

conn_str = (
    f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
    f"user={DB_USER} password={DB_PASSWORD}"
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
    Insert a row into the 'facecrime_data' table with a 768-d vector,
    storing everything in Postgres (pgvector).
    'filename' is used as the PK for uniqueness.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Insert the embedding as a vector(768)
                sql = """
                INSERT INTO facecrime_data
                    (filename, image_base64, embedding, sex, height, weight,
                     hairColor, eyeColor, race, sexOffender, offense, created_at)
                VALUES
                    (%s, %s, %s::vector(768), %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (filename) DO NOTHING
                """
                cur.execute(sql, (
                    filename,
                    image_base64,
                    embedding,
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
    Perform a similarity search using pgvector's <=> operator, but 
    we treat <=> as if it were a distance. We compute:
        matchPercent = 1 - (embedding <=> query_vector)
    and then ORDER BY matchPercent DESC for top matches.

    This yields a 0..1 range for matchPercent (clamped if out of range).
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # We do 1 - (embedding <=> ...) and call that match_percent
                # Then we sort in descending order so the highest match_percent is first
                sql = f"""
                SELECT
                  filename,
                  image_base64 AS image,
                  sex,
                  height,
                  weight,
                  hairColor,
                  eyeColor,
                  race,
                  sexOffender,
                  offense,
                  (1 - (embedding <=> %s::vector(768))) AS match_percent
                FROM facecrime_data
                ORDER BY (1 - (embedding <=> %s::vector(768))) DESC
                LIMIT {limit};
                """
                cur.execute(sql, (embedding, embedding))
                rows = cur.fetchall()

                results = []
                for row in rows:
                    raw_val = float(row["match_percent"])
                    # clamp to [0..1]
                    raw_val = max(0.0, min(1.0, raw_val))

                    results.append({
                        "filename": row["filename"],
                        "image": row["image"],  # base64
                        "sex": row["sex"],
                        "height": row["height"],
                        "weight": row["weight"],
                        "hairColor": row["haircolor"],
                        "eyeColor": row["eyecolor"],
                        "race": row["race"],
                        "sexOffender": row["sexoffender"],
                        "offense": row["offense"],
                        "matchPercent": raw_val
                    })
                return results

    except Exception as e:
        logger.error(f"Failed to query similar images: {e}")
        return []
