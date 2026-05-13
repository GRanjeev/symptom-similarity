"""
╔══════════════════════════════════════════════════════════════════╗
║           load_data.py — Rare Disease Data Loader               ║
║                                                                  ║
║  What this file does:                                            ║
║  • Reads disease_data.csv (100 rare diseases)                    ║
║  • Converts each disease profile into a 768-dimensional vector   ║
║  • Stores all vectors + metadata in Qdrant Cloud                 ║
║                                                                  ║
║  Run ONCE after generate_dataset.py:                             ║
║    python load_data.py                                           ║
║                                                                  ║
║  After this runs, the Qdrant collection is ready for             ║
║  similarity searches via search.py                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, pandas as pd
from dotenv import load_dotenv                    # Load API keys from .env
from sentence_transformers import SentenceTransformer  # Text → vector model
from qdrant_client import QdrantClient            # Qdrant vector DB client
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load QDRANT_URL and QDRANT_API_KEY from .env file
load_dotenv()

# ── Connect to Qdrant Cloud ───────────────────────────────────────
print('Connecting to Qdrant Cloud...')
client = QdrantClient(
    url=os.getenv('QDRANT_URL'),        # Your Qdrant cluster URL
    api_key=os.getenv('QDRANT_API_KEY'),  # Your Qdrant API key
)
print('Connected!')

# ── Load Embedding Model ──────────────────────────────────────────
# all-mpnet-base-v2 is a SentenceTransformer model that produces
# 768-dimensional vectors — a good balance of accuracy and speed.
# First run: downloads ~420MB model and caches locally.
# Subsequent runs: loads from cache in ~1 second.
print('Loading embedding model (~80MB download on first run)...')
model = SentenceTransformer('all-mpnet-base-v2')
print('Model ready!')

# ── Collection Configuration ──────────────────────────────────────
COLLECTION = 'rare_diseases'   # Name of the Qdrant collection
VECTOR_SIZE = 768              # all-mpnet-base-v2 output dimension


def create_collection():
    """
    Create the Qdrant collection for rare diseases.

    A Qdrant collection is like a table in a traditional database,
    but optimised for vector similarity search instead of exact lookups.

    Configuration:
    • size=768: Each vector has 768 dimensions (numbers)
    • distance=COSINE: Use cosine similarity to compare vectors
      (measures the angle between vectors, good for text)

    COSINE vs EUCLIDEAN:
    Cosine similarity works better for text embeddings because it
    measures the DIRECTION (meaning) of vectors, not their magnitude.
    Two sentences with similar meaning will have a small angle between
    their vectors, giving a high cosine similarity score.
    """
    # Check if collection already exists (e.g. from a previous run)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print('Deleting old collection...')  # Start fresh
        client.delete_collection(COLLECTION)

    # Create new collection with correct dimensions
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=768,                # Must match model output dimension
            distance=Distance.COSINE  # Cosine similarity for text vectors
        )
    )
    print(f'Created collection: {COLLECTION}')


def build_text(row) -> str:
    """
    Build the text that will be embedded for each disease.

    WHY WE REPEAT SYMPTOMS 3 TIMES:
    The embedding model creates a vector that represents the AVERAGE
    meaning of all words in the text. By repeating symptoms,
    we give them more weight in the final vector.
    This means searches based on symptoms will score higher —
    which is what we want since users describe symptoms.

    Args:
        row: A pandas DataFrame row (one disease from disease_data.csv)

    Returns:
        Rich text string combining all disease information
    """
    return f"""
{row['name']} {row['name']}
symptoms: {row['symptoms']}
patient presents with: {row['symptoms']}
clinical features include: {row['symptoms']}
category: {row['category']}
{row['description']}
    """.strip()


def load_diseases():
    """
    Read CSV, embed each disease, and upload all vectors to Qdrant.

    PROCESS FOR EACH DISEASE:
    1. Build rich text (disease name + symptoms × 3 + description)
    2. Pass text through SentenceTransformer → get 768-dim vector
    3. Create a PointStruct (vector + metadata payload)
    4. Batch upload all points to Qdrant

    WHY BATCH UPLOAD:
    Uploading all 100 points in one upsert() call is much faster
    than uploading them one at a time (1 network round trip vs 100).
    """
    # Load the CSV generated by generate_dataset.py
    df = pd.read_csv('disease_data.csv')
    print(f'Loaded {len(df)} diseases from disease_data.csv')

    points = []  # Collect all PointStructs before uploading

    for idx, row in df.iterrows():
        # STEP 1: Build the rich text for this disease
        text = build_text(row)

        # STEP 2: Convert text to a 768-dimensional vector
        # normalize_embeddings=True applies L2 normalisation
        # This ensures cosine similarity calculations are accurate
        vector = model.encode(text, normalize_embeddings=True).tolist()

        # STEP 3: Create a PointStruct
        # A PointStruct contains:
        # • id: unique identifier (we use the row index)
        # • vector: the 768-dim embedding (used for similarity search)
        # • payload: metadata dict (returned with search results)
        point = PointStruct(
            id=idx,        # Row index as ID (0, 1, 2, ...)
            vector=vector, # The 768-dim vector representation
            payload={
                # Payload is stored alongside the vector in Qdrant.
                # These fields are returned in search results (hit.payload)
                # and displayed in the dashboard result cards.
                'name':        str(row['name']),        # Disease name
                'category':    str(row['category']),    # e.g. "Neurological"
                'symptoms':    str(row['symptoms']),    # Comma-separated symptoms
                'description': str(row['description']), # Plain English description
                'specialist':  str(row['specialist']),  # e.g. "Neurologist"
                'prevalence':  str(row['prevalence']),  # e.g. "1 in 100000"
            }
        )
        points.append(point)
        print(f'  [{idx+1}/{len(df)}] Embedded: {row["name"]}')

    # STEP 4: Upload all points to Qdrant in a single batch
    # upsert() = insert or update (idempotent — safe to run multiple times)
    client.upsert(collection_name=COLLECTION, points=points)
    print(f'Uploaded {len(points)} vectors to Qdrant!')
    print('Upload complete! Check cloud.qdrant.io to verify.')


if __name__ == '__main__':
    # Run both steps in order
    create_collection()  # Create the Qdrant collection first
    load_diseases()      # Then embed and upload all diseases
    print()
    print('All done! Next: python search.py')
