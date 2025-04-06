import os
import logging
import datetime
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Build the Postgres connection string from environment
host = os.environ.get("MANUFACTURER_DB_HOST", "localhost")
port = os.environ.get("MANUFACTURER_DB_PORT", "5432")
dbname = os.environ.get("MANUFACTURER_DB_NAME", "pomudatabase")
user = os.environ.get("MANUFACTURER_DB_USER", "pomudatabaseuser")
password = os.environ.get("MANUFACTURER_DB_PASSWORD", "pomudatabasepass")

conn_str = f"host={host} port={port} dbname={dbname} user={user} password={password}"

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
                # If needed, convert embedding (Python list) to "[...]" string.
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
                    embedding,  # either use psycopg adaptation or convert to bracketed string
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
    Perform a similarity search using pgvector operator (<->) with COSINE distance.
    We convert distance to a 'matchPercent' by computing (1 - distance).
    That yields a value in [0..1] if vectors are normalized or roughly so.

    If the 'images' table is indexed with:
      CREATE INDEX ON images USING hnsw (embedding vector_cosine_ops)
      WITH (m=16, ef_construction=128);

    Then the <-> operator uses cosine distance (1 - dot_product).
    We'll convert it to similarity as matchPercent = 1 - distance.
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # We want to order by the raw distance (embedding <-> queryVector),
                # but also compute (1 - distance) as matchPercent
                # The smallest distance is the best match, so "ORDER BY embedding <-> %s::vector(768)" is correct.
                # Then we show (1 - distance) in the output.
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
                  (1 - (embedding <-> %s::vector(768))) AS matchPercent
                FROM images
                ORDER BY embedding <-> %s::vector(768)
                LIMIT {limit};
                """
                cur.execute(sql, (embedding, embedding))
                rows = cur.fetchall()

                results = []
                for row in rows:
                    # Convert to the structure your frontend expects
                    # In your example, "image" corresponds to "image_base64",
                    # "matchPercent" is a float in [0..1].
                    results.append({
                        "image": row["image_base64"],
                        "offense": row["offense"],
                        "height": row["height"],
                        "weight": row["weight"],
                        "hairColor": row["haircolor"],
                        "eyeColor": row["eyecolor"],
                        "race": row["race"],
                        "sexOffender": row["sexoffender"],
                        "matchPercent": float(row["matchpercent"]),
                    })
                return results
    except Exception as e:
        logger.error(f"Failed to query similar images: {e}")
        return []
