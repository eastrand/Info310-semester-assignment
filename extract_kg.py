import json
import asyncio
import os
import yaml
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def load_schema(schema_path: str) -> dict:
    """Load and parse the schema YAML file."""
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


def build_schema_prompt(schema: dict) -> str:
    """Build a prompt section describing the schema."""
    prompt_parts = []
    
    if "entities" in schema:
        entity_lines = []
        for entity_dict in schema["entities"]:
            for entity_type, description in entity_dict.items():
                entity_lines.append(f"- {entity_type}: {description}")
        prompt_parts.append(
            "Extract these entity types:\n" + "\n".join(entity_lines)
        )
    
    if "relationships" in schema:
        rel_lines = []
        for rel_dict in schema["relationships"]:
            for rel_type, rel_spec in rel_dict.items():
                rel_lines.append(f"- {rel_type}: {rel_spec}")
        prompt_parts.append(
            "\nExtract these relationship types:\n" + "\n".join(rel_lines)
        )
    
    return "\n".join(prompt_parts)


async def extract_kg(args):
    client = AsyncOpenAI(
        base_url=args.llm_base_url, 
        api_key=args.llm_api_key
    )
    driver = GraphDatabase.driver(
        args.neo4j_url, 
        auth=(args.neo4j_user, args.neo4j_password)
    )
    
    # Load schema
    schema = load_schema(args.schema_path)
    schema_prompt = build_schema_prompt(schema)
    
    # Setup progress tracking
    progress_file = Path(args.progress_file)
    completed_docs = set()
    
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
            completed_docs = set(progress_data.get("completed_docs", []))
        print(f"Resuming: {len(completed_docs)} documents already processed")
    
    with driver.session() as session:
        docs = session.run(
            "MATCH (d:Document) RETURN d.doc_id AS id, d.text AS text"
        ).data()
    
    # Filter out already completed documents
    docs_to_process = [d for d in docs if d["id"] not in completed_docs]
    print(f"Extracting KG from {len(docs_to_process)} remaining documents")
    
    async def process_doc(doc):
        system_prompt = (
            "You are an information extraction model. "
            "Return only valid JSON with two arrays: 'entities' and 'relationships'."
        )
        user_prompt = f"""Extract entities and relationships from this document according to the following schema:

{schema_prompt}

Return JSON in this exact format: {{"entities":[{{"type":"","name":"","properties":{{}}}}], "relationships":[{{"source":"","target":"","type":"","properties":{{}}}}]}}

Document text:
{doc['text'][:10000]}"""  # truncate if too long
        
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
            with driver.session() as session:
                for ent in parsed.get("entities", []):
                    # Create entity with type as label
                    entity_type = ent.get("type", "Entity")
                    properties = ent.get("properties", {})
                    properties["name"] = ent["name"]
                    
                    # Build property set clause
                    prop_items = ", ".join([f"e.{k} = ${k}" for k in properties.keys()])
                    
                    session.run(
                        f"MERGE (e:{entity_type} {{name: $name}}) "
                        f"SET {prop_items} "
                        "WITH e MATCH (d:Document {doc_id: $doc_id}) "
                        "MERGE (e)-[:DERIVED_FROM]->(d)",
                        **properties,
                        doc_id=doc["id"]
                    )
                
                for rel in parsed.get("relationships", []):
                    rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
                    properties = rel.get("properties", {})
                    
                    # Build property set clause for relationship
                    if properties:
                        prop_items = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
                        prop_set = f"SET {prop_items}"
                    else:
                        prop_set = ""
                    
                    session.run(
                        f"MATCH (s {{name: $src}}), (t {{name: $tgt}}) "
                        f"MERGE (s)-[r:{rel_type}]->(t) "
                        f"{prop_set}",
                        src=rel["source"], 
                        tgt=rel["target"],
                        **properties
                    )
            
            # Mark as completed and save progress
            completed_docs.add(doc["id"])
            with open(progress_file, "w") as f:
                json.dump({"completed_docs": list(completed_docs)}, f)
                
        except Exception as e:
            print(f"⚠️ Error processing {doc['id']}: {e}")
    
    # Process with semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(args.concurrent_requests)
    
    async def process_with_semaphore(doc):
        async with semaphore:
            await process_doc(doc)
    
    tasks = [process_with_semaphore(d) for d in docs_to_process]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="KG extraction"):
        await coro
    
    driver.close()
    print("✅ KG extraction complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-url", default=os.getenv("NEO4J_URL"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen/qwen3-4b-thinking-2507"))
    parser.add_argument("--schema-path", default="schema.yaml")
    parser.add_argument("--progress-file", default="kg_extraction_progress.json")
    parser.add_argument("--concurrent-requests", type=int, default=5)
    args = parser.parse_args()
    
    asyncio.run(extract_kg(args))