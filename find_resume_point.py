# save as: list_missing_run_starts.py
import os
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv

DOCUMENT_CACHE_PATH = "document_chunks.pkl"

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")        # e.g. neo4j+s://<id>.databases.neo4j.io
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

assert NEO4J_URL and NEO4J_USER and NEO4J_PASSWORD, "Missing Neo4j env vars."

def fetch_all_chunk_ids():
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    ids = set()
    with driver.session() as session:
        # Stream all chunkIds
        result = session.run("MATCH (c:Chunk) RETURN c.chunkId AS id")
        for rec in result:
            cid = rec["id"]
            if cid is not None:
                ids.add(str(cid))
    driver.close()
    return ids

def main():
    if not os.path.exists(DOCUMENT_CACHE_PATH):
        raise FileNotFoundError(f"Missing {DOCUMENT_CACHE_PATH}")

    with open(DOCUMENT_CACHE_PATH, "rb") as f:
        documents = pickle.load(f)

    db_ids = fetch_all_chunk_ids()

    def get_chunk_id(doc):
        meta = getattr(doc, "metadata", {}) or {}
        return meta.get("chunk_id")

    prev_present = True  # so if index 0 is missing, we print it
    endIndex = 0
    for i, doc in enumerate(documents):
        cid = get_chunk_id(doc)
        present = (cid in db_ids) if cid else False

        # Print only when this is a missing item AND previous was present
        if (not present) and prev_present:
            # Fall back label for missing metadata
            label = cid if cid else "<no_chunk_id>"
            print(f"{i}, {label}")

        prev_present = present  # advance
        if not present:
            endIndex = i
    
    print("end", endIndex)
    print(f"Last index: {endIndex}")
if __name__ == "__main__":
    main()
