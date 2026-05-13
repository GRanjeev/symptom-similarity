"""
╔══════════════════════════════════════════════════════════════════╗
║              search.py — Qdrant Rare Disease Search              ║
║                                                                  ║
║  What this file does:                                            ║
║  • Converts symptom text into a 768-dimensional vector           ║
║  • Searches the Qdrant "rare_diseases" collection                ║
║  • Returns the top 5 most similar diseases with match scores     ║
║                                                                  ║
║  WHY VECTOR SEARCH INSTEAD OF KEYWORD SEARCH:                    ║
║  Typing "muscle stiffness triggered by noise" into a keyword     ║
║  search returns nothing for "Stiff Person Syndrome".             ║
║  Vector search converts BOTH into mathematical representations   ║
║  and finds them to be 75%+ similar — because they MEAN          ║
║  the same thing, even though the words don't match.             ║
║                                                                  ║
║  Test: python search.py                                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
from dotenv import load_dotenv                    # Load API keys from .env
from sentence_transformers import SentenceTransformer  # Text → vector model
from qdrant_client import QdrantClient            # Qdrant vector DB client

# Load environment variables (QDRANT_URL, QDRANT_API_KEY) from .env file
load_dotenv()

# ── Lazy Singletons ───────────────────────────────────────────────
# We use module-level variables so the Qdrant connection and embedding
# model are created ONCE and reused on every search.
#
# WHY SINGLETONS:
# • SentenceTransformer downloads/loads an 80MB model from disk.
#   Loading it on every search would take ~3 seconds each time.
# • QdrantClient opens a network connection.
#   Creating it once is much faster than reconnecting each time.
# • Streamlit reruns the entire script on every interaction,
#   so without singletons, the model would reload constantly.
_client = None   # Qdrant connection — None until first search
_model  = None   # Embedding model — None until first search


def get_client():
    """
    Return the shared Qdrant client.
    Creates the connection on the first call, then reuses it.
    Reads QDRANT_URL and QDRANT_API_KEY from the .env file.
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv('QDRANT_URL'),        # e.g. https://xxxx.cloud.qdrant.io
            api_key=os.getenv('QDRANT_API_KEY'),  # Your Qdrant Cloud API key
        )
    return _client


def get_model():
    """
    Return the shared SentenceTransformer embedding model.
    Loads from disk on the first call (~80MB, ~2s), then reuses it.

    MODEL CHOICE: all-mpnet-base-v2
    • Produces 768-dimensional vectors (vs 384 for smaller models)
    • Better semantic understanding of medical/emotional language
    • Standard choice for high-quality semantic similarity tasks
    """
    global _model
    if _model is None:
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model


def search_diseases(symptom_text: str, top_k: int = 5) -> list[dict]:
    """
    Main search function — converts symptoms to vector, queries Qdrant.

    ┌─────────────────────────────────────────────────────────┐
    │ HOW VECTOR SEARCH WORKS (step by step):                  │
    │                                                          │
    │ 1. User types: "muscle stiffness triggered by noise"     │
    │                          ↓                               │
    │ 2. SentenceTransformer converts this to:                 │
    │    [0.023, -0.412, 0.891, 0.034, ...]  ← 768 numbers   │
    │    This vector captures the MEANING of the text          │
    │                          ↓                               │
    │ 3. Qdrant searches all stored disease vectors using      │
    │    cosine similarity — finds nearest neighbours          │
    │                          ↓                               │
    │ 4. Returns: "Stiff Person Syndrome" at 75.7%            │
    │    because its stored vector is mathematically close     │
    │    even though the exact words never appear              │
    └─────────────────────────────────────────────────────────┘

    Args:
        symptom_text: Plain English description of symptoms from the user
        top_k:        How many results to return (default: 5)

    Returns:
        List of dicts, each containing:
        - name, category, symptoms, description, specialist, prevalence
          (from the Qdrant payload stored when load_data.py was run)
        - similarity_pct: cosine similarity as a percentage (0-100)
        - match_level: "Strong Match" / "Possible Match" / "Weak Match"
        - color: hex colour for the bar chart
    """
    # Don't search if input is empty (user clicked button with no text)
    if not symptom_text.strip():
        return []

    # ── STEP 1: Convert symptom text to a vector ──────────────────
    # encode() runs the text through the neural network.
    # normalize_embeddings=True applies L2 normalisation so that
    # cosine similarity calculations are accurate.
    vector = get_model().encode(
        symptom_text,
        normalize_embeddings=True  # Required for correct cosine similarity
    ).tolist()  # Convert numpy array to Python list for Qdrant

    # ── STEP 2: Search Qdrant using the vector ────────────────────
    # query_points() performs Approximate Nearest Neighbour (ANN) search.
    # It finds the top_k vectors in the collection that are most
    # similar to our query vector, measured by cosine similarity.
    #
    # The collection "rare_diseases" was created and populated by
    # running: python load_data.py
    # It contains 100 disease profiles, each stored as a 768-dim vector.
    response = get_client().query_points(
        collection_name='rare_diseases',  # Our Qdrant collection name
        query=vector,                      # The 768-dim query vector
        limit=top_k,                      # Return only top 5 matches
        with_payload=True,                # Include stored metadata (name, etc.)
    )
    results = response.points  # List of ScoredPoint objects

    # ── STEP 3: Format results for the dashboard ──────────────────
    formatted = []
    for hit in results:
        # hit.score is the cosine similarity: 0.0 (no match) to 1.0 (identical)
        # We multiply by 100 to get a percentage that's easier to read
        pct = round(hit.score * 100, 1)

        # Classify the match level based on similarity percentage.
        # These thresholds were tuned based on testing with real symptoms.
        # Note: Medical semantic search rarely exceeds 80% — 55%+ is meaningful.
        if pct >= 55:
            level = 'Strong Match'
            color = '#DC2626'   # Red — high confidence match
        elif pct >= 40:
            level = 'Possible Match'
            color = '#D97706'   # Orange — moderate confidence
        else:
            level = 'Weak Match'
            color = '#16A34A'   # Green — low confidence, show for completeness

        # Build the result dict from the Qdrant payload.
        # hit.payload contains everything we stored in load_data.py:
        # name, category, symptoms, description, specialist, prevalence
        formatted.append({
            'name':           hit.payload.get('name', ''),
            'category':       hit.payload.get('category', ''),
            'symptoms':       hit.payload.get('symptoms', ''),
            'description':    hit.payload.get('description', ''),
            'specialist':     hit.payload.get('specialist', ''),
            'prevalence':     hit.payload.get('prevalence', ''),
            'similarity_pct': pct,     # e.g. 75.7
            'match_level':    level,   # e.g. "Strong Match"
            'color':          color,   # Hex color for bar chart
        })

    return formatted  # List sorted by similarity (Qdrant returns highest first)


# ── Quick Test ────────────────────────────────────────────────────
# Run this file directly to test the search without the dashboard:
#   python search.py
#
# Expected output for this test:
#   1. Stiff Person Syndrome — 75.7% (Strong Match)
#   2. Myasthenia Gravis    — 55.5% (Strong Match)
#   3. Polymyositis         — 51.0% (Possible Match)
if __name__ == '__main__':
    # Example: detailed symptoms of Stiff Person Syndrome
    test = """
progressive muscle stiffness, sudden painful muscle spasms triggered by
loud noise or sudden surprise, spasms triggered by emotional stress,
anxiety, difficulty walking, hunched posture, muscle rigidity in back
and legs, falls frequently, stiffness worsens over time
"""
    print(f'Testing Qdrant search with: {test.strip()[:80]}...')
    print()
    for i, r in enumerate(search_diseases(test), 1):
        print(f'{i}. {r["name"]} — {r["similarity_pct"]}% ({r["match_level"]})')
