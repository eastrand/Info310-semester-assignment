import os, json, requests, textwrap

# === CONFIG ===
LMSTUDIO_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-thinking-2507")
SCHEMA_PATH = "schema.yaml"
TEST_TEXT = open("test1.txt", "r", encoding="utf-8").read()[:2000]  # read first 2000 chars

# === Load schema and build prompt like main script ===
import yaml
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = yaml.safe_load(f)

entity_defs = "\n".join([f"- {list(item.keys())[0]}: {list(item.values())[0]}" for item in schema.get("entities", [])])
rel_defs = "\n".join([f"- {list(item.keys())[0]}: {list(item.values())[0]}" for item in schema.get("relationships", [])])
allowed_rel_types = [list(item.keys())[0] for item in schema.get("relationships", [])]

prompt = textwrap.dedent(f"""
You are an expert data analyst specializing in international law and peace agreements.
Extract entities and relationships in JSON format based on this schema.

Entities:
{entity_defs}

Relationships:
{rel_defs}

Allowed relationship types (must match exactly): {allowed_rel_types}

Document:
---
{TEST_TEXT}
---

Return ONLY a valid JSON object with two keys:
"entities" and "relationships".
""")

# === Send to LM Studio ===
# payload = {
#     "model": MODEL,
#     "messages": [
#         {"role": "system", "content": "You output only valid JSON objects. No explanations."},
#         {"role": "user", "content": prompt},
#     ],
#     "temperature": 0.0,
#     "max_tokens": 512,
# }
# payload = {
#     "model": MODEL,
#     "messages": [
#         {
#             "role": "system",
#             "content": (
#                 "You are a strict information extraction model. "
#                 "You must output ONLY valid JSON following the schema. "
#                 "No explanations, no <think> text, no prose."
#             ),
#         },
#         {"role": "user", "content": prompt},
#     ],
#     "temperature": 0.0,
#     "max_tokens": 1024,
#     "response_format": {
#         "type": "json_schema",
#         "json_schema": {
#             "name": "kg_schema",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "entities": {
#                         "type": "array",
#                         "items": {
#                             "type": "object",
#                             "properties": {
#                                 "label": {"type": "string"},
#                                 "name": {"type": "string"},
#                             },
#                             "required": ["label", "name"],
#                         },
#                     },
#                     "relationships": {
#                         "type": "array",
#                         "items": {
#                             "type": "object",
#                             "properties": {
#                                 "source": {"type": "string"},
#                                 "target": {"type": "string"},
#                                 "type": {"type": "string"},
#                             },
#                             "required": ["source", "target", "type"],
#                         },
#                     },
#                 },
#                 "required": ["entities", "relationships"],
#             },
#         },
#     },
# }

payload = {
    "model": MODEL,
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
Extract all entities and relationships from the following text,
based on the schema provided in your system prompt.

Text:
\"\"\"{TEST_TEXT[:2000]}\"\"\"  # truncated to 2000 chars for safety

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



print("Sending request to LM Studio...")
r = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
print(f"Status: {r.status_code}")
print()

try:
    data = r.json()
    output = data["choices"][0]["message"]["content"]
    print("=== Raw model output ===")
    print(output)
    print()

    # Try extracting JSON
    import re
    def extract_json(text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

    js = extract_json(output)
    if js:
        parsed = json.loads(js)
        print("=== Parsed JSON successfully ===")
        print(json.dumps(parsed, indent=2))
    else:
        print("No JSON object detected in model output.")
except Exception as e:
    print("Failed to parse model output:", e)
    print("Raw text was:")
    print(r.text)
