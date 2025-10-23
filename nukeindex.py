from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")
print("neo4jurl",NEO4J_URL)
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE","neo4j")

driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session(database=NEO4J_DB) as s:
    # see what's there
    idx = s.run("""
        SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options
        RETURN name, type, entityType, labelsOrTypes, properties, options
    """).data()
    print(idx)

    # drop the offending index (and any you want to reset)
    s.run("DROP INDEX docs_index IF EXISTS")
    s.run("DROP INDEX notdocs_index_1 IF EXISTS")   # optional, if you created it already
