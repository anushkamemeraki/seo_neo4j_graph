import os
import json
import time
import pandas as pd
import google.generativeai as genai

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

#configurations
CSV_PATH = "enriched_sheet.csv"
BATCH_SIZE = 20
MAX_WORKERS = 5

# POST-PROCESSING FILTERS
# The prompt stays generic — these deterministic rules clean up
# whatever noise slips through.

GENERIC_ARTFORMS = {
    "art", "arts", "artwork", "artworks", "paintings", "painting",
    "craft", "crafts", "craftsmanship", "artistry", "art forms",
    "artform", "artforms", "design", "designs", "narratives",
    "stories", "decor", "decorations", "creations", "expressions",
    "folk art", "tribal art", "indian art", "indian arts",
    "traditional art", "traditional arts", "handmade art",
    "indigenous art", "folk arts", "tribal arts", "visual art",
    "fine art", "fine arts", "handpainted paintings",
    "handmade paintings", "handmade artworks", "art traditions",
    "art tradition", "art prints","traditional art form", "indian art form",
    "historical art forms", "contemporary art forms", "arts and crafts"
}

NOISE_ARTISTS = {
    "memeraki", "master artists", "masters", "kaarigar",
    "artisans", "women artists", "japanese artists",
    "contemporary indian artists", "indian tribes", "puppeteers",
    "gond tribe", "meenas", "woodcarvers", "family of woodcarvers",
    "tharu women", "nomadic indian tribe", "house of edwa",
    "naina creation", "artists in jahangir's atelier",
    "meena artists", "baiga", "bhil", "kurumba",
}

NOISE_REGIONS = {
    "india", "indian", "world", "global south", "south asia",
    "local", "regional",
}

# LOAD & PREPARE DATA
# CHANGE 1: pass description to the model, not just slug

df = pd.read_csv(CSV_PATH)
df = df[df["Landing page type"].isin(["Collection", "Blog Article"])]
rows = []

for idx, row in df.iterrows():

    title       = str(row.get("title", "") or "").strip()
    path        = str(row.get("Landing page path", "") or "").strip()
    description = str(row.get("description", "") or "").strip()
    slug        = path.split("/")[-1].replace("-", " ").strip()

    if description:
        context_text = f"Title: {title}\nDescription: {description[:1500]}"
    else:
        context_text = (
            f"Title: {title}\n"
            f"Slug: {slug}\n"
            f"Note: No description available. Only extract what the title clearly states."
        )

    rows.append({
        "id":      idx,
        "path":    path,
        "context": context_text,
    })

batches = []
for i in range(0, len(rows), BATCH_SIZE):
    batches.append(rows[i:i + BATCH_SIZE])

# PROMPT — kept generic, no domain-specific examples
# CHANGE 2: removed all specificity; let the model reason freely

SYSTEM_PROMPT = """
Extract these fields from every item:

- artist  : named individual people who created the work
- artform : specific named art or craft traditions
- region  : specific geographic locations (avoid country-level only)
- theme   : subjects, motifs, narratives, or concepts depicted

Rules:
- Only extract entities explicitly mentioned or strongly implied.
- Do not invent information.
- If unknown, return empty arrays.

Return STRICT JSON ARRAY ONLY. No markdown, no explanation, no extra text.
"""

# POST-PROCESSING
# CHANGE 3: noise filtering happens in code, not in the prompt

def clean_entities(record: dict) -> dict:

    def filter_artforms(items):
        return [
            a for a in items
            if a.lower().strip() not in GENERIC_ARTFORMS
        ]

    def filter_artists(items):
        return [
            a for a in items
            if a.lower().strip() not in NOISE_ARTISTS
            and len(a.strip().split()) >= 2
        ]

    def filter_regions(items):
        return [
            r for r in items
            if r.lower().strip() not in NOISE_REGIONS
        ]

    record["artform"] = filter_artforms(record.get("artform", []))
    record["artist"]  = filter_artists(record.get("artist", []))
    record["region"]  = filter_regions(record.get("region", []))

    return record


# PROCESS FUNCTION

def process_batch(batch_index: int, batch: list) -> list:

    batch_input = [
        {"id": item["id"], "text": item["context"]}
        for item in batch
    ]

    prompt = f"{SYSTEM_PROMPT}\n\nINPUT:\n{json.dumps(batch_input, indent=2, ensure_ascii=False)}"

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        cleaned = [clean_entities(r) for r in parsed]
        print(f" Batch {batch_index} done ({len(cleaned)} records)")
        return cleaned

    except Exception as e:
        print(f" Batch {batch_index} failed: {e}")
        return []


results = []
start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(process_batch, idx, batch)
        for idx, batch in enumerate(batches)
    ]
    for future in as_completed(futures):
        results.extend(future.result())

total = len(results)
fully_empty = sum(
    1 for r in results
    if not any([r.get("artist"), r.get("artform"), r.get("region"), r.get("theme")])
)
print(f"\n Quality report:")
print(f"  Total: {total}")
print(f"  Fully empty: {fully_empty} ({fully_empty/total*100:.1f}%)")
print(f"  Has artform: {sum(1 for r in results if r.get('artform'))}")
print(f"  Has artist:  {sum(1 for r in results if r.get('artist'))}")
print(f"  Has region:  {sum(1 for r in results if r.get('region'))}")

with open("extracted_entities_v3.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

end = time.time()
print(f"\n Saved extracted_entities_v3.json")
print(f" Time: {round(end - start, 2)}s")