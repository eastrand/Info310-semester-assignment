import aiohttp
import asyncio
import json
from tqdm import tqdm
from utils import log

async def extract_from_text(session, text, model, url):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert information extraction system. "
                    "You analyze input text and output a structured JSON object "
                    "that follows the given schema. Do not output <think> or reasoning text."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract all entities and relationships from the following text.

Text:
\"\"\"{text[:6000]}\"\"\"

Each entity must have:
- a label (type from the schema, e.g., Agreement, Party, Person, etc.)
- a name (the actual string from the text)

Each relationship must have:
- a source (entity name)
- a target (entity name)
- a type (relationship from the schema)

Return only valid JSON following the schema, without explanations or blank arrays.
"""
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "kg_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "name": {"type": "string"}
                                },
                                "required": ["label", "name"]
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "type": {"type": "string"}
                                },
                                "required": ["source", "target", "type"]
                            }
                        }
                    },
                    "required": ["entities", "relationships"]
                }
            }
        }
    }

    try:
        async with session.post(url, json=payload, timeout=120) as resp:
            if resp.status != 200:
                log(f"⚠️ LLM error ({resp.status}): {await resp.text()[:200]}")
                return None
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        log(f"⚠️ Error extracting KG: {e}")
        return None


async def extract_knowledge_graph_async(documents, args):
    LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
    MODEL = args.llm_model or "qwen/qwen3-4b-thinking-2507"

    async with aiohttp.ClientSession() as session:
        tasks = [extract_from_text(session, doc.page_content, MODEL, LMSTUDIO_URL)
                 for doc in documents[: args.max_docs]]
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting KG"):
            r = await future
            if r:
                results.append(r)
        return results


def extract_knowledge_graph(documents, args):
    log("Starting Knowledge Graph extraction phase…")
    results = asyncio.run(extract_knowledge_graph_async(documents, args))
    with open("kg_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"✅ Extracted {len(results)} KG outputs. Saved to kg_results.json")
