"""
╔══════════════════════════════════════════════════════════════════╗
║           ai_explainer.py — Groq AI Explanation Module          ║
║                                                                  ║
║  What this file does:                                            ║
║  • Takes Qdrant search results + original symptom text           ║
║  • Sends them to Groq (Llama 3) via API                          ║
║  • Returns a plain-English explanation of why symptoms match     ║
║                                                                  ║
║  WHY WE NEED THIS:                                               ║
║  Qdrant tells us WHAT matches (Stiff Person Syndrome: 75%).      ║
║  Groq explains WHY it matches in human language, what the        ║
║  condition is, and what the patient should do next.              ║
║  Together they are much more useful than either alone.           ║
║                                                                  ║
║  API USED: Groq Cloud (free) running Meta's Llama 3 model        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
from groq import Groq                # Groq Python SDK
from dotenv import load_dotenv       # Load API keys from .env file

# Load GROQ_API_KEY from .env into environment variables
load_dotenv()

# ── Lazy Singleton ────────────────────────────────────────────────
# Groq client is created once and reused.
# Streamlit reruns the script on every interaction —
# using a singleton avoids creating a new API client each time.
_client = None


def get_client():
    """
    Return the shared Groq client.
    Creates it on the first call using GROQ_API_KEY from .env.
    """
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return _client


def explain_results(symptom_text: str, results: list[dict]) -> str:
    """
    Generate a plain-English explanation of Qdrant search results.

    ┌─────────────────────────────────────────────────────────┐
    │ HOW THIS WORKS:                                          │
    │                                                          │
    │ Input:  User symptoms + top 3 Qdrant matches             │
    │                   ↓                                      │
    │ Prompt: Carefully crafted to produce a warm,             │
    │         concise, accurate response                       │
    │                   ↓                                      │
    │ Groq:   Llama 3 (70B parameters) generates response      │
    │                   ↓                                      │
    │ Output: 120-word explanation with:                       │
    │         • Why symptoms suggest the top match             │
    │         • Which specialist to see                        │
    │         • Reassurance + disclaimer                       │
    └─────────────────────────────────────────────────────────┘

    WHY GROQ INSTEAD OF OPENAI:
    Groq provides Llama 3 (Meta's open-source model) completely
    free with generous rate limits — no credit card required.
    For a hackathon, this means zero API costs.

    Args:
        symptom_text: The original symptom description from the user
        results:      List of formatted dicts from search_diseases()
                      (contains name, description, similarity_pct, etc.)

    Returns:
        Plain-English explanation string, or error message if API fails
    """
    if not results:
        return 'No matching conditions found. Please try describing symptoms differently.'

    # ── BUILD THE PROMPT ──────────────────────────────────────────
    # We only send the top 3 matches to keep the prompt concise.
    # More context doesn't always mean better AI output —
    # focused prompts with clear instructions produce better responses.
    top3 = results[:3]

    # Format the matches as a readable list for the AI
    matches_text = '\n'.join([
        f'- {r["name"]} ({r["similarity_pct"]}% match): {r["description"]}'
        for r in top3
    ])

    # The prompt is carefully designed to:
    # 1. Give the AI the right context (medical info assistant)
    # 2. Provide structured input (symptoms + matches)
    # 3. Specify exactly what to output (numbered requirements)
    # 4. Set hard constraints (length, disclaimer, tone)
    prompt = f"""You are a compassionate medical information assistant.
A patient described their symptoms and a vector similarity search found matching conditions.

Patient symptoms: {symptom_text}

Top matching conditions found:
{matches_text}

Write a response that includes:
1. One sentence explaining which symptoms most strongly suggest the top match.
2. Which specialist the patient should see.
3. One warm and reassuring sentence.

Important rules:
- Maximum 120 words.
- You MUST include: This is NOT a medical diagnosis.
- Use plain English. No medical jargon.
- Be warm, clear, and supportive."""

    # ── CALL GROQ API ──────────────────────────────────────────────
    try:
        response = get_client().chat.completions.create(
            model='llama-3.3-70b-versatile',  # Free Groq model (70B parameters)
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=0.4,   # Slightly creative but mostly consistent
                               # 0.0 = deterministic, 1.0 = very random
            max_tokens=200,    # ~120 words — keeps response concise
        )

        # Extract the text response from the API response object
        return response.choices[0].message.content

    except Exception as e:
        # Graceful degradation — if API fails, show a helpful message
        # rather than crashing the entire app
        return f'AI explanation unavailable ({e}). Please review results above.'
