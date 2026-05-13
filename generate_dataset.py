"""
╔══════════════════════════════════════════════════════════════════╗
║        generate_dataset.py — Rare Disease Dataset Generator     ║
║                                                                  ║
║  What this file does:                                            ║
║  • Uses Groq (Llama 3) to generate structured profiles for       ║
║    100 rare diseases across 10 medical categories                ║
║  • Saves results to disease_data.csv                             ║
║                                                                  ║
║  WHY WE GENERATE INSTEAD OF USING AN EXISTING DATASET:          ║
║  • Rare disease databases are proprietary and expensive          ║
║  • We need SYMPTOM-RICH descriptions in plain English            ║
║    (not ICD codes or clinical terminology)                       ║
║  • Groq Llama 3 produces medically accurate, consistently        ║
║    formatted data in the exact structure Qdrant needs            ║
║                                                                  ║
║  Run ONCE: python generate_dataset.py                            ║
║  Output:   disease_data.csv (100 diseases)                       ║
║  Next:     python load_data.py                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, csv, json, time
from groq import Groq                # Groq Python SDK for Llama 3 API
from dotenv import load_dotenv       # Load API keys from .env file

# Load GROQ_API_KEY from .env file
load_dotenv()

# Connect to Groq API — completely free with generous rate limits
# Model used: llama-3.3-70b-versatile (Meta's Llama 3, 70B parameters)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ── 100 Rare Diseases (10 categories × 10 diseases) ──────────────
# Organised into 10 clinical categories to ensure broad coverage.
# This means the Qdrant collection covers a wide symptom space —
# important for demonstrating that vector search finds the right
# disease regardless of which body system is affected.
DISEASE_GROUPS = [
    # Autoimmune — immune system attacks the body's own tissues
    ['Systemic Lupus Erythematosus', 'Sjogren Syndrome',
     'Mixed Connective Tissue Disease', 'Dermatomyositis',
     'Polymyositis', 'Behcet Disease', 'Goodpasture Syndrome',
     'Myasthenia Gravis', 'Pemphigus Vulgaris', 'Antiphospholipid Syndrome'],

    # Neurological — affecting the nervous system
    ['Stiff Person Syndrome', 'Huntington Disease', 'Wilson Disease',
     'Friedreich Ataxia', 'Guillain-Barre Syndrome', 'Narcolepsy',
     'Kleine-Levin Syndrome', 'Rasmussen Encephalitis',
     'Moyamoya Disease', 'Cavernous Malformation'],

    # Metabolic — enzyme deficiencies and metabolic pathway disorders
    ['Phenylketonuria', 'Maple Syrup Urine Disease', 'Gaucher Disease',
     'Fabry Disease', 'Pompe Disease', 'Niemann-Pick Disease',
     'Mucopolysaccharidosis', 'Porphyria', 'Galactosemia', 'Homocystinuria'],

    # Genetic — chromosomal and single-gene disorders
    ['Marfan Syndrome', 'Ehlers-Danlos Syndrome', 'Williams Syndrome',
     'Angelman Syndrome', 'Prader-Willi Syndrome', 'Fragile X Syndrome',
     'Turner Syndrome', 'Klinefelter Syndrome', 'DiGeorge Syndrome', 'Rett Syndrome'],

    # Hematological — blood and bone marrow disorders
    ['Aplastic Anemia', 'Paroxysmal Nocturnal Hemoglobinuria',
     'Thrombotic Thrombocytopenic Purpura', 'Hemophilia A',
     'Hemophilia B', 'Von Willebrand Disease', 'Sickle Cell Disease',
     'Beta Thalassemia', 'Diamond-Blackfan Anemia', 'Polycythemia Vera'],

    # Respiratory — lungs and airways
    ['Cystic Fibrosis', 'Alpha-1 Antitrypsin Deficiency',
     'Pulmonary Arterial Hypertension', 'Lymphangioleiomyomatosis',
     'Sarcoidosis', 'Hypersensitivity Pneumonitis',
     'Bronchiolitis Obliterans', 'Alveolar Proteinosis',
     'Pulmonary Langerhans Cell Histiocytosis', 'Pleuritis'],

    # Skin — dermatological rare diseases
    ['Epidermolysis Bullosa', 'Ichthyosis', 'Mastocytosis',
     'Pemphigoid', 'Pityriasis Rubra Pilaris', 'Netherton Syndrome',
     'Incontinentia Pigmenti', 'Darier Disease',
     'Hailey-Hailey Disease', 'Erythropoietic Protoporphyria'],

    # Musculoskeletal — bones, joints, and connective tissue
    ['Osteogenesis Imperfecta', 'Achondroplasia',
     'Fibrodysplasia Ossificans Progressiva', 'McCune-Albright Syndrome',
     'Morquio Syndrome', 'Alkaptonuria', 'Pseudoxanthoma Elasticum',
     'Hypoparathyroidism', 'Hypophosphatasia', 'Gorham Disease'],

    # Endocrine — hormonal and glandular disorders
    ['Addison Disease', 'Cushing Syndrome', 'Acromegaly',
     'Hyperaldosteronism', 'Multiple Endocrine Neoplasia Type 1',
     'Congenital Adrenal Hyperplasia', 'Carcinoid Syndrome',
     'Phaeochromocytoma', 'Insulinoma', 'Autoimmune Thyroiditis'],

    # Gastrointestinal — digestive system and liver
    ['Primary Sclerosing Cholangitis', 'Autoimmune Hepatitis',
     'Primary Biliary Cholangitis', 'Eosinophilic Esophagitis',
     'Intestinal Lymphangiectasia', 'Short Bowel Syndrome',
     'Hirschsprung Disease', 'Whipple Disease',
     'Menetrier Disease', 'IPEX Syndrome'],
]


def generate_disease_entry(disease_name: str) -> dict:
    """
    Ask Groq Llama 3 to describe one rare disease as structured JSON.

    PROMPT DESIGN STRATEGY:
    The prompt is carefully structured to produce data that works
    well with vector similarity search:

    1. "symptoms" field: 8-12 descriptors in PLAIN ENGLISH
       (not ICD codes or medical abbreviations)
       → Users will search in plain English, so we embed in plain English

    2. "description": 2 sentences for non-doctors
       → Explains the condition without jargon

    3. "specialist": Who to see
       → Actionable next step shown in the dashboard

    4. "prevalence": How rare
       → Puts the disease in context (e.g. "1 in 100000 people")

    WHY JSON FORMAT:
    Structured JSON means we can reliably extract fields and store
    them in the Qdrant payload for later retrieval in search results.

    Args:
        disease_name: The name of the rare disease to describe

    Returns:
        Dict with keys: name, category, symptoms, description,
        specialist, prevalence
    """
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',   # Free Groq model
        messages=[{
            'role': 'user',
            'content': f'''Describe the rare disease "{disease_name}" for a medical database.
Return ONLY a valid JSON object. No markdown. No code blocks. No extra text. Just JSON:
{{
  \"name\": \"exact disease name\",
  \"category\": \"one of: Autoimmune/Neurological/Metabolic/Genetic/Hematological/Respiratory/Skin/Musculoskeletal/Endocrine/Gastrointestinal\",
  \"symptoms\": \"comma-separated list of 8-12 key symptoms in plain English\",
  \"description\": \"2 sentence plain English description for non-doctors\",
  \"specialist\": \"which doctor to see e.g. Rheumatologist\",
  \"prevalence\": \"how rare e.g. 1 in 100000 people\"
}}'''
        }],
        temperature=0.3,   # Low temperature = more factual, less creative
                           # Medical data should be accurate and consistent
        max_tokens=400,    # Enough for all fields, prevents excessive output
    )

    text = response.choices[0].message.content.strip()

    # Clean response — Llama sometimes wraps JSON in markdown code blocks
    # even when we ask it not to. This strips any ```json ... ``` wrapping.
    text = text.replace('```json', '').replace('```', '').strip()

    # Parse the cleaned text as JSON
    # If this fails, a JSONDecodeError is raised and caught in the main loop
    return json.loads(text)


def save_to_csv(diseases: list, filename: str = 'disease_data.csv'):
    """
    Save the generated disease list to a CSV file.

    WHY CSV:
    • Human-readable — judges can inspect the data directly
    • Easy to load with pandas in load_data.py
    • Version-controllable in git (unlike binary formats)
    • Can be edited manually to add or correct entries

    Args:
        diseases: List of disease dicts from generate_disease_entry()
        filename: Output CSV filename (default: disease_data.csv)
    """
    # Define column order — same order used in load_data.py and search results
    fieldnames = ['name', 'category', 'symptoms', 'description',
                  'specialist', 'prevalence']

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()    # Write column names as first row
        writer.writerows(diseases)  # Write all disease rows

    print(f'Saved {len(diseases)} diseases to {filename}')


if __name__ == '__main__':
    print('Generating dataset with Groq (Llama 3) — free and fast!')
    print()

    diseases = []
    total = sum(len(g) for g in DISEASE_GROUPS)  # 100 total
    count = 0

    # Iterate through each category group, then each disease in the group
    for group in DISEASE_GROUPS:
        for disease_name in group:
            count += 1
            print(f'  [{count}/{total}] {disease_name}', end=' ... ', flush=True)

            try:
                entry = generate_disease_entry(disease_name)

                # Validate that all required fields are present
                # A partial entry would cause errors in load_data.py
                required = ['name', 'category', 'symptoms',
                            'description', 'specialist', 'prevalence']
                if all(k in entry for k in required):
                    diseases.append(entry)
                    print('OK')
                else:
                    missing = [k for k in required if k not in entry]
                    print(f'Skipped: missing fields {missing}')

            except json.JSONDecodeError:
                # Groq occasionally returns malformed JSON.
                # We skip and continue rather than crashing.
                print('Skipped: JSON parse error')

            except Exception as e:
                # Network errors, rate limits, etc.
                print(f'Skipped: {e}')

            # Small delay between API calls — polite to Groq's servers
            # and avoids hitting rate limits on the free tier
            time.sleep(2)

    print()
    save_to_csv(diseases)
    print(f'Done! {len(diseases)}/{total} diseases generated.')
    print('Next step: python load_data.py')
