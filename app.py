"""
╔══════════════════════════════════════════════════════════════════╗
║              SymptomSimilarity — Main Application               ║
║                                                                  ║
║  What this file does:                                            ║
║  • Renders the Streamlit web dashboard                           ║
║  • Tab 1: Rare Disease Search (vector similarity via Qdrant)     ║
║  • Tab 2: Mental Health Support (separate Qdrant collection)     ║
║  • Crisis detection: shows helplines before any search results   ║
║                                                                  ║
║  Run: streamlit run app.py                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st          # Web dashboard framework
import plotly.graph_objects as go  # Interactive charts (bar, radar, gauge)
import os
from dotenv import load_dotenv                    # Load API keys from .env file
from sentence_transformers import SentenceTransformer  # Text → vector embeddings
from qdrant_client import QdrantClient            # Qdrant vector database client
from search import search_diseases                # Our Qdrant rare disease search
from ai_explainer import explain_results          # Our Groq AI explanation module

# Load all API keys from the .env file into environment variables
load_dotenv()

# Configure the Streamlit page — title, icon, and wide layout for charts
st.set_page_config(
    page_title='SymptomSimilarity',
    layout='wide'
)

# ═════════════════════════════════════════════════════════════
# SECTION 1: CRISIS DETECTION
# ─────────────────────────────────────────────────────────────
# Mental health safety feature:
# If the user types any of these crisis-related phrases,
# the app immediately stops and shows crisis helplines.
# No disease results are shown until the person is safe.
# ═════════════════════════════════════════════════════════════

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life",
    "don't want to live", "want to die", "no reason to live",
    "not worth living", "better off dead", "self harm",
    "hurt myself", "cut myself", "overdose", "no point",
    "can't go on", "give up on life", "end it all",
]

# Verified free crisis helplines — India and International
CRISIS_RESOURCES = {
    "iCall (India)":         "9152987821",           # Mon-Sat 8am-10pm
    "Vandrevala Foundation": "1860-2662-345 (24/7)", # 24/7 available
    "NIMHANS Bangalore":     "080-46110007",          # Government hospital
    "iMHANS Delhi":          "011-40770770",          # Government service
    "Befrienders India":     "044-24640050",          # Chennai-based
}

INTERNATIONAL = {
    "Crisis Text Line":      "Text HOME to 741741",  # USA/Canada/UK
    "Befrienders Worldwide": "befrienders.org",       # Global directory
    "IASP":                  "https://www.iasp.info/resources/Crisis_Centres/",
}


def detect_crisis(text: str) -> bool:
    """
    Scan the user's input for crisis-related language.

    HOW IT WORKS:
    Converts text to lowercase, then checks if any crisis keyword
    appears anywhere in the text. Returns True if crisis detected.

    WHY THIS MATTERS:
    Mental health search could be used by people in genuine distress.
    We check BEFORE running any Qdrant search — crisis help comes first.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)


def show_crisis_resources():
    """
    Display crisis helplines in a prominent red box.

    Called when detect_crisis() returns True.
    After this, st.stop() prevents any further code from running —
    the user sees ONLY this, not any disease similarity results.
    """
    st.markdown("""
<div style="background:#FEE2E2;border:3px solid #DC2626;border-radius:12px;
            padding:28px;margin:12px 0;">
<h2 style="color:#DC2626;margin:0 0 8px 0;">🆘 You Are Not Alone</h2>
<p style="font-size:17px;color:#374151;margin:0 0 20px 0;">
What you are feeling matters. Please reach out to someone right now.
These services are <strong>free, confidential, and available immediately.</strong>
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Crisis Helplines — India")
    for name, number in CRISIS_RESOURCES.items():
        st.markdown(f"**{name}:** `{number}`")

    st.markdown("---")
    st.markdown("### International Resources")
    for name, contact in INTERNATIONAL.items():
        st.markdown(f"**{name}:** {contact}")

    st.markdown("---")
    st.info(
        "If you are in immediate danger, please call your local emergency "
        "services (112 in India) or go to your nearest hospital emergency department."
    )


# ═════════════════════════════════════════════════════════════
# SECTION 2: MENTAL HEALTH VECTOR SEARCH
# ─────────────────────────────────────────────────────────────
# This section searches a SEPARATE Qdrant collection called
# "mental_health_profiles" which contains 55+ conditions
# described using emotional language (feelings, thoughts,
# physical sensations) rather than clinical symptoms.
#
# WHY A SEPARATE COLLECTION:
# Rare diseases and mental health need different embeddings.
# "I feel empty" should match depression, not a skin condition.
# ═════════════════════════════════════════════════════════════

# Lazy singletons — loaded once on first use, then reused
# This avoids reloading the 80MB model on every Streamlit rerun
_mh_client = None   # Qdrant connection for mental health collection
_mh_model  = None   # SentenceTransformer model for embeddings


def get_mh_client():
    """Return shared Qdrant client (creates it on first call only)."""
    global _mh_client
    if _mh_client is None:
        _mh_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),       # From .env file
            api_key=os.getenv("QDRANT_API_KEY"),  # From .env file
        )
    return _mh_client


def get_mh_model():
    """Return shared embedding model (loads from disk on first call only)."""
    global _mh_model
    if _mh_model is None:
        # all-mpnet-base-v2 produces 768-dimensional vectors
        # Better semantic understanding than the smaller MiniLM model
        _mh_model = SentenceTransformer("all-mpnet-base-v2")
    return _mh_model


def search_mental_health(feeling_text: str, top_k: int = 5) -> list:
    """
    Core mental health vector search using Qdrant.

    HOW IT WORKS:
    1. Convert the user's feelings description into a 768-dim vector
    2. Search the "mental_health_profiles" Qdrant collection
    3. Qdrant returns the top_k most similar profiles (cosine similarity)
    4. Format results with urgency levels and match classifications

    WHAT MAKES THIS DIFFERENT FROM KEYWORD SEARCH:
    Typing "I feel empty and nothing matters" would find ZERO results
    with keyword search. Vector search finds Major Depressive Disorder
    because the mathematical meaning of those words is close to the
    emotional descriptors we stored for that condition.

    Args:
        feeling_text: User's free-text description of how they feel
        top_k:        How many results to return (default 5)

    Returns:
        List of dicts containing condition info + similarity scores
    """
    if not feeling_text.strip():
        return []  # Don't search if input is empty

    try:
        # STEP 1: Convert text to vector
        # The embedding model maps text to a point in 768-dimensional space.
        # Similar meanings → nearby points → high cosine similarity
        vector = get_mh_model().encode(
            feeling_text,
            normalize_embeddings=True  # L2 normalize so cosine similarity works correctly
        ).tolist()

        # STEP 2: Search Qdrant
        # query_points() finds the nearest vectors in the collection
        # using cosine similarity (configured when collection was created)
        results = get_mh_client().query_points(
            collection_name="mental_health_profiles",  # Our mental health collection
            query=vector,                               # The 768-dim query vector
            limit=top_k,                               # Return top 5 matches
            with_payload=True,                         # Include stored metadata
        ).points

        # STEP 3: Format results for the dashboard
        formatted = []
        for hit in results:
            pct     = round(hit.score * 100, 1)  # Convert 0-1 similarity to percentage
            urgency = hit.payload.get("urgency", "consider-support")

            # Map urgency level to display label and colour
            if urgency == "seek-help-now":
                urgency_label = "🔴 Please seek help now"
                urgency_color = "#DC2626"    # Red
            elif urgency == "speak-to-professional-soon":
                urgency_label = "🟡 Consider speaking to a professional"
                urgency_color = "#D97706"    # Orange
            else:
                urgency_label = "🟢 Self-care and monitoring"
                urgency_color = "#16A34A"    # Green

            # Map similarity percentage to match level and bar colour
            if pct >= 65:
                match_level = "Strong Match"
                color       = "#DC2626"      # Red bar
            elif pct >= 45:
                match_level = "Possible Match"
                color       = "#D97706"      # Orange bar
            else:
                match_level = "Weak Match"
                color       = "#16A34A"      # Green bar

            # Build the formatted result dictionary
            formatted.append({
                "name":              hit.payload.get("name", ""),
                "category":          hit.payload.get("category", ""),
                "feelings":          hit.payload.get("feelings", ""),
                "thoughts":          hit.payload.get("thoughts", ""),
                "physical":          hit.payload.get("physical", ""),
                "coping":            hit.payload.get("coping", ""),
                "professional_help": hit.payload.get("professional_help", ""),
                "urgency":           urgency,
                "urgency_label":     urgency_label,
                "urgency_color":     urgency_color,
                "description":       hit.payload.get("description", ""),
                "similarity_pct":    pct,
                "match_level":       match_level,
                "color":             color,
            })
        return formatted

    except Exception as e:
        st.error(f"Mental health search error: {e}")
        return []


def get_mh_ai_response(feeling_text: str, results: list) -> str:
    """
    Generate a warm, compassionate AI response using Groq (Llama 3).

    HOW IT WORKS:
    Takes the user's description + top 3 Qdrant matches,
    sends them to Groq Llama 3 with a carefully crafted prompt,
    and returns a human-friendly explanation.

    WHY THIS MATTERS:
    Qdrant finds the WHAT (which condition matches).
    Groq explains the WHY in warm, non-clinical language.
    Together they bridge the gap between a similarity score
    and something a person can actually understand and act on.

    Args:
        feeling_text: Original user input
        results:      List of Qdrant search results (formatted dicts)

    Returns:
        String containing the AI-generated compassionate response
    """
    from groq import Groq
    if not results:
        return ""
    try:
        # Summarise top 3 matches for the AI prompt
        top3 = results[:3]
        matches = "\n".join([
            f"- {r['name']} ({r['similarity_pct']}%): {r['description']}"
            for r in top3
        ])

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free Groq model
            messages=[{"role": "user", "content": f"""
Someone described how they are feeling: "{feeling_text}"

The closest matching conditions from our database are:
{matches}

Write a warm compassionate response that:
1. Acknowledges what they are feeling (1 sentence)
2. Gently explains what the closest match suggests (1-2 sentences)
3. Recommends the most important next step (1 sentence)
4. Ends with one sentence of genuine encouragement

Rules:
- Maximum 100 words
- NEVER clinical or cold
- ALWAYS include: This is not a diagnosis — please speak to a mental health professional
- Warm and human like a caring friend
- No bullet points — flowing sentences only
"""}],
            temperature=0.4,   # Slightly creative but consistent
            max_tokens=200,    # Short enough to read quickly
        )
        return resp.choices[0].message.content.strip()
    except:
        return ""


def show_mental_health_tab():
    """
    Render the complete Mental Health Support tab.

    Flow:
    1. Show info banner and always-visible crisis resources
    2. Text input for feelings description
    3. On search:
       a. Crisis check → if triggered, show helplines and STOP
       b. Qdrant vector search on mental_health_profiles collection
       c. Groq AI generates compassionate response
       d. Show: urgency banner, charts, result cards, coping strategies
       e. Always show crisis lines at the bottom
    """
    # Info banner at top of mental health tab
    st.markdown("""
<div style="background:linear-gradient(135deg,#EDE9FE,#DBEAFE);
            border-radius:12px;padding:20px;margin-bottom:16px;">
<h3 style="color:#7C3AED;margin:0 0 6px 0;">Mental Health Support</h3>
<p style="color:#374151;margin:0;font-size:15px;">
Describe how you are feeling in your own words — no clinical terms needed.
Qdrant will find the closest matching emotional patterns and suggest support.
</p>
</div>
""", unsafe_allow_html=True)

    # Crisis resources — always accessible even before searching
    # Collapsed by default to not overwhelm, but always one click away
    with st.expander(
        "🆘 Having thoughts of suicide or self-harm? Click here immediately",
        expanded=False
    ):
        for name, number in CRISIS_RESOURCES.items():
            st.markdown(f"**{name}:** `{number}`")
        st.markdown("**Crisis Text Line:** Text HOME to 741741")
        st.markdown("**Befrienders Worldwide:** befrienders.org")

    st.markdown("---")

    # Free-text input — no checkboxes, no forms, just plain language
    feeling_text = st.text_area(
        "How are you feeling right now?",
        height=150,
        placeholder=(
            "Write freely... e.g. I feel empty and nothing makes me happy anymore. "
            "I can't get out of bed. I feel like a burden to everyone around me. "
            "I don't know what's wrong with me but I can't stop crying..."
        ),
        key="mh_input"  # Unique key to avoid Streamlit element conflicts
    )

    search_btn = st.button(
        "Find Support & Understanding",
        type="primary",
        use_container_width=True,
        key="mh_search"
    )

    if search_btn and feeling_text.strip():

        # ── CRISIS CHECK — Always runs BEFORE any Qdrant search ──
        # If crisis keywords detected, show helplines and halt execution.
        # st.stop() prevents any further code from running.
        if detect_crisis(feeling_text):
            show_crisis_resources()
            st.stop()  # CRITICAL: nothing else runs after this

        # ── QDRANT MENTAL HEALTH SEARCH ───────────────────────────
        with st.spinner("Finding similar emotional patterns in Qdrant..."):
            results = search_mental_health(feeling_text, top_k=5)

        if not results:
            st.error(
                "Could not find matches. Please make sure the mental health "
                "collection is loaded: python load_mental_health.py"
            )
            st.stop()

        # ── GROQ AI COMPASSIONATE RESPONSE ────────────────────────
        with st.spinner("Groq is preparing a compassionate response..."):
            ai_response = get_mh_ai_response(feeling_text, results)

        # Display AI response in a purple highlighted box
        if ai_response:
            st.markdown(f"""
<div style="background:#EDE9FE;border-left:4px solid #7C3AED;
            border-radius:8px;padding:16px;margin:12px 0;">
<strong style="color:#7C3AED;">Compassionate AI Response</strong><br><br>
{ai_response}
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── URGENCY BANNER ─────────────────────────────────────────
        # The top result's urgency level determines the banner colour:
        # Red = seek help now, Yellow = see professional, Green = self-care
        top = results[0]
        urgency_bg = {
            "seek-help-now":              "#FEE2E2",  # Light red
            "speak-to-professional-soon": "#FEF3C7",  # Light yellow
            "consider-support":           "#DCFCE7",  # Light green
        }.get(top["urgency"], "#F3F4F6")

        st.markdown(f"""
<div style="background:{urgency_bg};border-radius:10px;
            padding:16px;margin:8px 0;text-align:center;">
<h3 style="margin:0;color:#374151;">{top['urgency_label']}</h3>
<p style="margin:4px 0 0 0;color:#6B7280;">
See a <strong>{top['professional_help']}</strong>
</p>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── VISUALISATION CHARTS ───────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.subheader("Similarity to Known Patterns")
            # Horizontal bar chart showing similarity % for each match
            # Red bars = strong match (≥65%), Orange = possible (≥45%), Green = weak
            names  = [r["name"][:28]+"..." if len(r["name"])>28 else r["name"] for r in results]
            scores = [r["similarity_pct"] for r in results]
            colors = [r["color"] for r in results]

            bar = go.Figure(go.Bar(
                x=scores, y=names, orientation="h",
                marker_color=colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
            ))
            bar.update_layout(
                xaxis=dict(range=[0, 115], title="Similarity %"),
                yaxis=dict(autorange="reversed"),   # Top result at top
                height=300,
                margin=dict(l=10, r=70, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",        # Transparent background
            )
            # key= is REQUIRED when multiple plotly charts exist on same page
            st.plotly_chart(bar, use_container_width=True, key="mh_bar")

        with right:
            st.subheader("Pattern Distribution")
            # Radar/spider chart shows how the top 5 matches compare
            # Each axis = one condition, filled area = similarity score
            cats = [r["name"][:16] for r in results]
            vals = [r["similarity_pct"] for r in results]
            # Close the radar loop by repeating the first point
            cats.append(cats[0]); vals.append(vals[0])

            radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                fillcolor="rgba(124,58,237,0.2)",   # Light purple fill
                line_color="#7C3AED",               # Purple border
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=300,
                margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(radar, use_container_width=True, key="mh_radar")

        st.divider()

        # ── RESULT CARDS ───────────────────────────────────────────
        # One expandable card per matched condition.
        # First card is expanded by default (highest similarity match).
        st.subheader("What These Patterns Suggest")
        for i, r in enumerate(results):
            with st.expander(
                f"{i+1}. {r['name']}  —  {r['similarity_pct']}%  {r['match_level']}",
                expanded=(i == 0)   # Auto-expand the top match
            ):
                col1, col2 = st.columns(2)

                with col1:
                    # Human-readable information from the Qdrant payload
                    st.markdown(f"**Category:** {r['category']}")
                    st.markdown(f"**About:** {r['description']}")
                    st.markdown(f"**Common feelings:** {r['feelings'][:200]}")
                    st.markdown(f"**Common thoughts:** {r['thoughts'][:200]}")
                    st.markdown(f"**Physical sensations:** {r['physical'][:200]}")

                with col2:
                    # Gauge chart showing match percentage as a speedometer
                    # Green zone = weak match, Yellow = possible, Red = strong
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r["similarity_pct"],
                        title={"text": "Pattern Match %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": r["color"]},
                            "steps": [
                                {"range": [0,  45], "color": "#F0FDF4"},   # Light green
                                {"range": [45, 65], "color": "#FEF3C7"},   # Light yellow
                                {"range": [65,100], "color": "#FEE2E2"},   # Light red
                            ]
                        }
                    ))
                    gauge.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=0)
                    )
                    # key must be unique — using loop index i ensures this
                    st.plotly_chart(
                        gauge,
                        use_container_width=True,
                        key=f"mh_gauge_{i}"  # e.g. mh_gauge_0, mh_gauge_1, ...
                    )
                    st.markdown(f"**See a:** {r['professional_help']}")
                    st.markdown(f"**Urgency:** {r['urgency_label']}")

                # Evidence-based coping strategies from the dataset
                st.markdown("**What has helped others with similar feelings:**")
                coping_list = [c.strip() for c in r["coping"].split(",")]
                for c in coping_list[:6]:
                    if c:
                        st.markdown(f"  ✅ {c}")

        st.divider()

        # ── HELPLINES ALWAYS AT BOTTOM OF RESULTS ─────────────────
        # Regardless of the search result, crisis lines are always shown.
        # A person may not be in crisis when they start but become distressed
        # reading about conditions — this ensures help is always visible.
        st.markdown("""
<div style="background:#EDE9FE;border-radius:10px;padding:16px;margin:8px 0;">
<h4 style="color:#7C3AED;margin:0 0 8px 0;">
Remember — You Don't Have to Face This Alone
</h4>
<p style="color:#374151;margin:0;">
Speaking to a mental health professional is a sign of strength, not weakness.
</p>
</div>
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**India Crisis Lines:**")
            for name, number in CRISIS_RESOURCES.items():
                st.markdown(f"- {name}: `{number}`")
        with col2:
            st.markdown("**International:**")
            for name, contact in INTERNATIONAL.items():
                st.markdown(f"- {name}: {contact}")

    elif search_btn:
        st.warning("Please describe how you are feeling first.")


# ═════════════════════════════════════════════════════════════
# SECTION 3: MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────
# Streamlit renders top to bottom on every interaction.
# The two-tab layout separates rare disease search from
# mental health support cleanly without URL routing.
# ═════════════════════════════════════════════════════════════

st.title("SymptomSimilarity")
st.caption(
    "300 million people live with rare diseases. "
    "1 in 4 people struggle with mental health. "
    "Vector similarity finds the closest matching conditions instantly."
)
# Global disclaimer — always visible regardless of which tab is active
st.warning("⚠️ FOR EDUCATIONAL PURPOSES ONLY — Not a medical diagnosis.")
st.divider()

# Create two tabs — Streamlit renders only the active tab's content
tab1, tab2 = st.tabs(["Rare Disease Search", "Mental Health Support"])


# ═════════════════════════════════════════════════════════════
# TAB 1: RARE DISEASE SEARCH
# ─────────────────────────────────────────────────────────────
# Uses the "rare_diseases" Qdrant collection (100 conditions).
# Embedding is based on symptom descriptions + disease name.
# search_diseases() in search.py handles all Qdrant logic.
# ═════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Describe Symptoms")

    # Free-text input — user describes symptoms in plain English
    # No drop-downs, no checkboxes — just natural language
    symptoms = st.text_area(
        "Enter symptoms in plain English:",
        height=150,
        placeholder=(
            "e.g. extreme fatigue, butterfly rash on face, joint pain, "
            "hair loss, sensitivity to sunlight"
        ),
        key="disease_input"  # Unique key prevents conflict with mh_input
    )

    search_clicked = st.button(
        "Find Similar Conditions",
        type="primary",
        use_container_width=True,
        key="disease_search"
    )
    st.divider()

    if search_clicked and symptoms.strip():

        # ── QDRANT VECTOR SEARCH ───────────────────────────────────
        # search_diseases() embeds the symptom text and queries Qdrant.
        # Returns top 5 matches with similarity scores from the
        # "rare_diseases" collection (100 diseases, 768-dim vectors).
        with st.spinner("Searching Qdrant vector database..."):
            results = search_diseases(symptoms, top_k=5)

        if not results:
            st.error("No results found. Try describing symptoms differently.")
            st.stop()

        # ── GROQ AI EXPLANATION ────────────────────────────────────
        # explain_results() sends the top 3 matches + symptom text
        # to Groq Llama 3 which generates a plain-English explanation.
        with st.spinner("Groq (Llama 3) is analysing the matches..."):
            explanation = explain_results(symptoms, results)

        st.subheader("AI Analysis (Groq Llama 3)")
        st.info(explanation)
        st.divider()

        # ── CHARTS: Bar + Radar side by side ──────────────────────
        left, right = st.columns(2)

        with left:
            st.subheader("Similarity Scores")
            # Truncate long disease names for display
            names  = [r["name"][:28]+"..." if len(r["name"])>28 else r["name"] for r in results]
            scores = [r["similarity_pct"] for r in results]
            colors = [r["color"] for r in results]  # Red/Orange/Green by match level

            bar = go.Figure(go.Bar(
                x=scores, y=names, orientation="h",
                marker_color=colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
            ))
            # Threshold lines show where Strong/Possible match boundaries are
            bar.add_vline(x=70, line_dash="dash", line_color="#DC2626", annotation_text="Strong")
            bar.add_vline(x=50, line_dash="dot",  line_color="#D97706", annotation_text="Possible")
            bar.update_layout(
                xaxis=dict(range=[0, 115], title="Similarity %"),
                yaxis=dict(autorange="reversed"),  # Highest score at top
                height=300,
                margin=dict(l=10, r=70, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # "rd_bar" key distinguishes this from the mental health bar chart
            st.plotly_chart(bar, use_container_width=True, key="rd_bar")

        with right:
            st.subheader("Match Distribution")
            cats = [r["name"][:18] for r in results]
            vals = [r["similarity_pct"] for r in results]
            cats.append(cats[0]); vals.append(vals[0])  # Close the radar loop

            radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                fillcolor="rgba(124,58,237,0.2)",
                line_color="#7C3AED",
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=300,
                margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(radar, use_container_width=True, key="rd_radar")

        st.divider()

        # ── RESULT CARDS ───────────────────────────────────────────
        # Expandable cards for each of the top 5 matched diseases.
        # Left column: disease info from Qdrant payload.
        # Right column: gauge chart showing match percentage.
        st.subheader("Detailed Results")
        for i, r in enumerate(results):
            with st.expander(
                f'{i+1}. {r["name"]}  —  {r["similarity_pct"]}%  {r["match_level"]}',
                expanded=(i == 0)  # Top match auto-expanded
            ):
                col1, col2 = st.columns(2)
                with col1:
                    # All data comes from the Qdrant payload
                    # (stored when we ran load_data.py)
                    st.markdown(f'**Category:** {r["category"]}')
                    st.markdown(f'**Prevalence:** {r["prevalence"]}')
                    st.markdown(f'**See a:** {r["specialist"]}')
                    st.markdown(f'**About:** {r["description"]}')
                    st.markdown(f'**Known symptoms:** {r["symptoms"]}')
                with col2:
                    # Gauge chart — visual representation of similarity score
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r["similarity_pct"],
                        title={"text": "Match %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": r["color"]},
                            "steps": [
                                {"range": [0,  50], "color": "#F0FDF4"},
                                {"range": [50, 70], "color": "#FEF3C7"},
                                {"range": [70,100], "color": "#FEE2E2"},
                            ]
                        }
                    ))
                    gauge.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=0)
                    )
                    # rd_gauge_0, rd_gauge_1, etc. — unique key per loop iteration
                    st.plotly_chart(
                        gauge,
                        use_container_width=True,
                        key=f"rd_gauge_{i}"
                    )

    elif search_clicked:
        st.warning("Please enter symptoms first.")
    else:
        st.info("Type symptoms above then click Search.")


# ═════════════════════════════════════════════════════════════
# TAB 2: MENTAL HEALTH SUPPORT
# ─────────────────────────────────────────────────────────────
# All mental health logic lives in show_mental_health_tab()
# defined above. Keeping it in a function makes the code
# readable and separates the two search flows cleanly.
# ═════════════════════════════════════════════════════════════
with tab2:
    show_mental_health_tab()


# Footer — always shown at the bottom of both tabs
st.divider()
st.caption(
    "Built with Qdrant (vector search) + Groq Llama 3 (AI) + Streamlit  |  "
    "Rare Disease Search + Mental Health Support"
)
