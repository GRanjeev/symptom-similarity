"""
╔══════════════════════════════════════════════════════════════════╗
║       load_mental_health.py — Mental Health Data Loader         ║
║                                                                  ║
║  What this file does:                                            ║
║  • Reads mental_health_data.csv (55 conditions)                  ║
║  • Converts each condition into a 768-dimensional vector         ║
║  • Stores all vectors in a SEPARATE Qdrant collection            ║
║                                                                  ║
║  WHY A SEPARATE COLLECTION FROM RARE DISEASES:                   ║
║  Mental health conditions are described using emotional          ║
║  language (feelings, thoughts, physical sensations) rather       ║
║  than clinical symptoms. Keeping them separate ensures           ║
║  "I feel empty" matches depression, not a skin condition.        ║
║                                                                  ║
║  Run ONCE after generate_mental_health.py:                       ║
║    python load_mental_health.py                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, pandas as pd
from dotenv import load_dotenv                    # Load API keys from .env
from sentence_transformers import SentenceTransformer  # Text → vector model
from qdrant_client import QdrantClient            # Qdrant vector DB client
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load QDRANT_URL and QDRANT_API_KEY from .env file
load_dotenv()

print("Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
print("Connected!")

# ── Same embedding model as the rare disease loader ───────────────
# Using the same model ensures consistent vector space across both
# collections. If the models differed, similarity scores would
# not be comparable between the two search tabs.
print("Loading embedding model (same as disease search)...")
model = SentenceTransformer("all-mpnet-base-v2")
print("Model ready!")

# ── Different collection name from rare diseases ──────────────────
COLLECTION  = "mental_health_profiles"  # SEPARATE from "rare_diseases"
VECTOR_SIZE = 768                       # Same dimension (all-mpnet-base-v2)


def create_collection():
    """
    Create the Qdrant collection for mental health profiles.

    Same configuration as the rare diseases collection:
    • 768-dimensional vectors (all-mpnet-base-v2 output)
    • Cosine similarity distance metric

    If the collection already exists (e.g. from a previous run),
    it is deleted and recreated fresh. This ensures clean data.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print("Collection exists. Deleting and recreating...")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,          # 768 dimensions
            distance=Distance.COSINE   # Cosine similarity for text
        )
    )
    print(f"Created collection: {COLLECTION}")


def build_text(row) -> str:
    """
    Build the text to embed for each mental health condition.

    KEY DESIGN DECISION — FEELINGS REPEATED TWICE:
    Unlike the disease loader (which repeats symptoms 3 times),
    here we repeat "feelings" twice. This is because:

    1. Users describe their FEELINGS ("I feel empty, hopeless")
       not their clinical symptoms
    2. By doubling the feelings text, these words dominate the
       embedding vector — so a search for "feeling empty and hopeless"
       will strongly match conditions with similar feeling descriptors
    3. "thoughts" and "physical" add context but don't dominate,
       which is correct since users rarely describe thought patterns
       in detail in a first search

    Args:
        row: A pandas DataFrame row (one mental health condition)

    Returns:
        Rich text string optimised for emotional similarity matching
    """
    return f"""
Mental health condition: {row['name']}
Category: {row['category']}
How this feels emotionally: {row['feelings']}
How this feels emotionally: {row['feelings']}
Common thought patterns: {row['thoughts']}
Physical sensations: {row['physical']}
Description: {row['description']}
    """.strip()


def load_data():
    """
    Read mental_health_data.csv, embed each condition, upload to Qdrant.

    THE DATA STRUCTURE:
    Each row in mental_health_data.csv has these columns:
    • name:              Condition name (e.g. "Major Depressive Disorder")
    • category:          e.g. "Depressive", "Anxiety", "Trauma"
    • feelings:          Plain English emotional descriptors (10-15 items)
    • thoughts:          Common thought patterns (5-8 items)
    • physical:          Body sensations and physical symptoms (5-8 items)
    • coping:            Evidence-based coping strategies (5-7 items)
    • professional_help: Which mental health professional to see
    • urgency:           "seek-help-now" / "speak-to-professional-soon" /
                         "consider-support"
    • description:       2-sentence plain English description

    All fields are stored in the Qdrant payload so they can be
    retrieved and displayed in search results without re-fetching.
    """
    df = pd.read_csv("mental_health_data.csv")
    print(f"Loaded {len(df)} conditions from mental_health_data.csv")
    print()

    points = []  # Collect all PointStructs before batch uploading

    for idx, row in df.iterrows():
        # Build the embedding text (feelings-heavy for emotional matching)
        text   = build_text(row)

        # Convert to 768-dimensional normalised vector
        vector = model.encode(text, normalize_embeddings=True).tolist()

        # Create Qdrant point with full metadata payload
        point = PointStruct(
            id=idx,        # Row index as unique ID
            vector=vector, # 768-dim embedding for similarity search
            payload={
                # All fields stored here are accessible in search results.
                # app.py reads these from hit.payload in search_mental_health()
                "name":              str(row["name"]),
                "category":          str(row["category"]),
                "feelings":          str(row["feelings"]),    # Shown in result cards
                "thoughts":          str(row["thoughts"]),    # Shown in result cards
                "physical":          str(row["physical"]),    # Shown in result cards
                "coping":            str(row["coping"]),      # Shown as ✅ list
                "professional_help": str(row["professional_help"]),  # "See a: ..."
                "urgency":           str(row["urgency"]),     # Determines banner color
                "description":       str(row["description"]), # Shown in result cards
            }
        )
        points.append(point)
        print(f"  [{idx+1}/{len(df)}] Embedded: {row['name']}")

    # Batch upload all vectors to Qdrant
    client.upsert(collection_name=COLLECTION, points=points)
    print()
    print(f"Uploaded {len(points)} vectors to Qdrant!")
    print()

    # Verify the upload succeeded by checking collection info
    info = client.get_collection(COLLECTION)
    print(f"Verified: {info.points_count} vectors in collection '{COLLECTION}'")
    print()
    print("All done! Next step: streamlit run app.py")


if __name__ == "__main__":
    create_collection()  # Create collection first
    load_data()          # Then embed and upload all conditions
