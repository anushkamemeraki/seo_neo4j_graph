import os
import json
import pandas as pd

from neo4j import GraphDatabase
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# LOAD ENV

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


df = pd.read_csv("enriched_sheet.csv")
df = df.reset_index(drop=True)

# LOAD ENTITY JSON

with open("extracted_entities_v3.json") as f:
    extracted = json.load(f)

entity_map = {}

for idx, item in enumerate(extracted):

    if "id" in item:
        entity_map[item["id"]] = item
    else:
        entity_map[idx] = item

# PROCESS SINGLE ROW

def process_row(idx, row):

    title = str(row.get("title", ""))
    path = str(row.get("Landing page path", ""))

    entities = entity_map.get(idx)

    if not entities:
        return

    node_type = "Collection"

    if "/blogs/" in path:
        node_type = "Blog"

    with driver.session() as session:

        # MAIN NODE

        session.run(
            f"""
            MERGE (n:{node_type} {{path: $path}})
            SET n.title = $title
            """,
            path=path,
            title=title
        )

        # ARTISTS

        for artist in entities.get("artist", []):

            session.run(
                f"""
                MATCH (n:{node_type} {{path: $path}})
                MERGE (a:Artist {{name: $artist}})
                MERGE (n)-[:BY_ARTIST]->(a)
                """,
                path=path,
                artist=artist
            )

        # ARTFORMS

        for artform in entities.get("artform", []):

            session.run(
                f"""
                MATCH (n:{node_type} {{path: $path}})
                MERGE (a:ArtForm {{name: $artform}})
                MERGE (n)-[:HAS_ARTFORM]->(a)
                """,
                path=path,
                artform=artform
            )

        # REGIONS

        for region in entities.get("region", []):

            session.run(
                f"""
                MATCH (n:{node_type} {{path: $path}})
                MERGE (r:Region {{name: $region}})
                MERGE (n)-[:FROM_REGION]->(r)
                """,
                path=path,
                region=region
            )

        # THEMES

        for theme in entities.get("theme", []):

            session.run(
                f"""
                MATCH (n:{node_type} {{path: $path}})
                MERGE (t:Theme {{name: $theme}})
                MERGE (n)-[:HAS_THEME]->(t)
                """,
                path=path,
                theme=theme
            )

    print(f"Done: {title}")

# MULTITHREADING

MAX_WORKERS = 10

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = []

    for idx, row in df.iterrows():

        futures.append(
            executor.submit(process_row, idx, row)
        )

    for future in as_completed(futures):
        future.result()

# CREATE SIMILARITY EDGES

with driver.session() as session:

    # COLLECTION ↔ COLLECTION

    session.run("""
    MATCH (c1:Collection)-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]->(x)
          <-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]-(c2:Collection)

    WHERE elementId(c1) < elementId(c2)

    MERGE (c1)-[:SIMILAR_COLLECTION]->(c2)
    """)

    # BLOG ↔ BLOG

    session.run("""
    MATCH (b1:Blog)-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]->(x)
          <-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]-(b2:Blog)

    WHERE elementId(b1) < elementId(b2)

    MERGE (b1)-[:SIMILAR_BLOG]->(b2)
    """)

    # BLOG ↔ COLLECTION

    session.run("""
    MATCH (b:Blog)-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]->(x)
          <-[:HAS_ARTFORM|BY_ARTIST|FROM_REGION|HAS_THEME]-(c:Collection)

    MERGE (b)-[:RELATED_COLLECTION]->(c)
    """)

print("\nKnowledge graph created!")

driver.close()

