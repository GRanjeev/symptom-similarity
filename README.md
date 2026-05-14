# SymptomSimilarity

Built for the Qdrant Hackathon 2026.

---

## What this is

A web app that lets you describe your symptoms or how you're feeling in plain English and finds the closest matching medical conditions using vector search.

There are two parts to it. The first is a rare disease explorer. The second is a mental health support tool.

Both of them work the same way under the hood- you type something, Qdrant searches for the nearest matching vectors, and the results come back ranked by similarity. The difference is what's stored in each collection and how the results are presented.

---

## Why I built it this way

The obvious approach for something like this would be a chatbot. Ask the user questions, collect structured answers, look things up in a database. I didn't do that because I think it misses the point.

The actual problem with rare disease diagnosis isn't that there's no information, it's that there's no connection between how patients describe their experience and how conditions are documented. A patient says "my muscles lock up when I get startled." A database has "Stiff Person Syndrome." Those two things don't share a keyword. Keyword search fails. Vector search doesn't, because it works on meaning rather than exact words.

The same thing applies to mental health. Most people who are struggling don't know what their condition is called. They know how they feel. So the mental health search is built around feeling descriptions- plain emotional language, not clinical criteria.

---

## The stack

- Qdrant Cloud (free tier) for vector storage and search
- sentence-transformers (all-mpnet-base-v2) to convert text to 768-dimensional vectors
- Groq running Llama 3 (llama-3.3-70b-versatile) for AI explanations — also free
- Streamlit for the frontend
- Python 3.11

No paid APIs. Everything runs on free tiers.

---

## Setup

You need Python 3.11, a Qdrant Cloud account (free at cloud.qdrant.io), and a Groq API key (free at console.groq.com).

```bash
git clone https://github.com/YOUR_USERNAME/symptom-similarity.git
cd symptom-similarity

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

---

## Loading the data

There are two Qdrant collections. You need to load both before the app will work.

**Rare diseases** — this calls the Groq API to generate profiles for 100 conditions across 10 medical categories. Takes about 5 minutes:

```bash
python generate_dataset.py
python load_data.py
```

**Mental health** — this one is hardcoded, no API needed, runs instantly:

```bash
python generate_mental_health.py
python load_mental_health.py
```

After both finish you'll have `rare_diseases` (100 vectors) and `mental_health_profiles` (68 vectors) in your Qdrant cluster.

---

## Running it

```bash
streamlit run app.py
```

Goes up at http://localhost:8501.

---

## How the search actually works

When you type something, the sentence-transformers model converts your text to a 768-dimensional vector. Qdrant runs cosine similarity search against all stored vectors and returns the closest matches. That's it.

The part that matters is what we store. For rare diseases, the embedding text is built around symptoms — the symptom description is repeated three times in the text that gets embedded, so that symptom-heavy queries score higher than generic ones. For mental health, the feelings field is repeated twice because that's what users describe when they're struggling.

The embedding model (all-mpnet-base-v2) handles the rest. It was chosen over smaller alternatives because 768-dimensional vectors give better separation between conditions that have overlapping vocabulary.

---

## Mental health and crisis safety

The mental health tab scans every input for crisis-related language before running any search. If something matches words like "suicide," "want to die," "end my life", the search stops entirely and the app shows crisis resources. Nothing else. No similarity scores, no condition names.

Crisis helplines shown in the app:

India:
- iCall: 9152987821 (Mon–Sat, 8am–10pm)
- Vandrevala Foundation: 1860-2662-345 (24/7)
- NIMHANS Bangalore: 080-46110007
- iMHANS Delhi: 011-40770770
- Befrienders India: 044-24640050

International:
- Crisis Text Line: text HOME to 741741
- Befrienders Worldwide: befrienders.org

These are also shown at the bottom of every mental health result regardless of what was searched. The idea is that someone might not be in crisis when they start a search but could become distressed reading about conditions.

---

## Files

```
app.py                    Streamlit app, both tabs, crisis detection
search.py                 Qdrant search for rare diseases
ai_explainer.py           Groq integration for explanations
generate_dataset.py       Generates disease_data.csv via Groq API
load_data.py              Embeds and uploads rare diseases to Qdrant
generate_mental_health.py Creates mental_health_data.csv (hardcoded)
load_mental_health.py     Embeds and uploads mental health data to Qdrant
disease_data.csv          100 rare disease profiles
mental_health_data.csv    55 mental health condition profiles
requirements.txt          Dependencies
.env                      Your API keys
```

---

## Dependencies

```
qdrant-client==1.7.3
sentence-transformers==2.3.1
groq==0.9.0
streamlit==1.29.0
plotly==5.18.0
pandas==2.1.4
python-dotenv==1.0.0
httpx==0.27.0
```

---

## Things to try

Rare disease tab:

```
progressive muscle stiffness, painful spasms that get triggered by loud
noises or sudden surprises, gets worse when I'm stressed or emotional,
difficulty walking, my back feels constantly rigid
```

Should come back with Stiff Person Syndrome

```
butterfly rash across my cheeks, constant fatigue, joint pain, hair
falling out, skin reacts badly to sunlight, low grade fever that won't
go away
```

Should match Systemic Lupus Erythematosus.

Mental health tab:

```
I feel completely empty. Nothing makes me happy anymore and I don't
know why. I can't get out of bed most days. I feel like I'm just a
burden to everyone around me.
```

Should match Major Depressive Disorder

---

## Hackathon category

Community and Social Impact.

---

## Disclaimer

This is for educational purposes only. Not a medical diagnosis tool. If you're in a medical or mental health emergency, please contact a professional or use the crisis resources listed above.
