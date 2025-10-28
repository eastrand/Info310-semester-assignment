import os
import re
import pickle
import gc
import json
import requests
import asyncio, time
import httpx
from typing import Optional, Tuple
from langchain_core.documents import Document  # keeps your cached type usable
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv
from json_repair import repair_json

# NEW: llama-cpp-python
# pip install llama-cpp-python
# from llama_cpp import Llama
load_dotenv()



CONCURRENCY = int(os.getenv("EXTRACT_CONCURRENCY", "12"))   # try 4–12; tune up/down
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5


def normalize_whitespace(s: str) -> str:
    # collapse tabs/newlines/extra spaces
    return re.sub(r'\s+', ' ', s).strip()

def normalize_structured(data: dict) -> dict:
    """Coerce model output into a strict schema:
       entities: list of {label: str, name: str}
       relationships: list of {source: str, target: str, type: str}
       - Expands list-valued names/sources/targets into multiple rows.
       - Accepts entities as strings.
    """
    entities_out = []
    rels_out = []

    # --- Entities ---
    for ent in (data.get("entities") or []):
        if isinstance(ent, str):
            names = [ent]
            label = "Entity"
        elif isinstance(ent, dict):
            label = ent.get("label", "Entity")
            names_raw = ent.get("name", "")
            if isinstance(names_raw, list):
                names = [str(x) for x in names_raw]
            elif isinstance(names_raw, (str, int, float)):
                names = [str(names_raw)]
            else:
                continue
        else:
            continue

        label = sanitize_label(label)
        for n in names:
            n = normalize_whitespace(str(n))
            if n:
                entities_out.append({"label": label, "name": n})

    # --- Relationships ---
    for rel in (data.get("relationships") or []):
        if not isinstance(rel, dict):
            continue
        src_raw = rel.get("source", "")
        tgt_raw = rel.get("target", "")
        typ_raw = rel.get("type", "RELATED_TO")

        # allow list-valued ends; expand cartesian if both are lists
        srcs = src_raw if isinstance(src_raw, list) else [src_raw]
        tgts = tgt_raw if isinstance(tgt_raw, list) else [tgt_raw]
        rel_type = sanitize_rel_type(str(typ_raw))

        for s in srcs:
            for t in tgts:
                s = normalize_whitespace(str(s))
                t = normalize_whitespace(str(t))
                if s and t:
                    rels_out.append({"source": s, "target": t, "type": rel_type})

    return {"entities": entities_out, "relationships": rels_out}

def build_payload(text: str):
    user_prompt = extraction_prompt_template.format(chunk_text=text) + \
                  "\nReturn only a single minified JSON object. No prose, no backticks."
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": "You output only valid JSON objects. No explanations."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False
    }
    payload_with_rf = dict(payload)
    payload_with_rf["response_format"] = {"type": "json_object"}
    return payload_with_rf, payload

async def infer_json_async(client: httpx.AsyncClient, text: str) -> str:
    payload_rf, payload_plain = build_payload(text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = await client.post(LMSTUDIO_API_URL, json=payload_rf, timeout=REQUEST_TIMEOUT)
            if res.status_code == 400:
                res = await client.post(LMSTUDIO_API_URL, json=payload_plain, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == MAX_RETRIES:
                return ""
            await asyncio.sleep(RETRY_BACKOFF ** attempt)

async def process_batch_async(batch, driver):
    limits = httpx.Limits(max_keepalive_connections=CONCURRENCY, max_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(doc):
            async with sem:
                return await infer_json_async(client, doc.page_content)

        tasks = [asyncio.create_task(one(doc)) for doc in batch]
        outputs = await asyncio.gather(*tasks)

    # write results
    with driver.session() as session:
        for doc, output in zip(batch, outputs):
            if not output:
                continue
            js = extract_first_json_object(output)
            if not js:
                # one targeted retry with reminder
                retry_text = doc.page_content + "\nRemember: Output must be a single compact JSON object."
                # do a quick sync fallback using requests (or you could call infer_json_async via asyncio.run)
                try:
                    payload_rf, payload_plain = build_payload(retry_text)
                    r = requests.post(LMSTUDIO_API_URL, json=payload_rf, timeout=REQUEST_TIMEOUT)
                    if r.status_code == 400:
                        r = requests.post(LMSTUDIO_API_URL, json=payload_plain, timeout=REQUEST_TIMEOUT)
                    r.raise_for_status()
                    data = r.json()
                    output = data["choices"][0]["message"]["content"]
                    js = extract_first_json_object(output)
                    if not js:
                        continue
                except Exception as e:
                    print(f"Exception", e)
                    continue
            try:
                parsed = json.loads(js)
                parsed = normalize_structured(parsed)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {js}")
                continue

            session.execute_write(
                write_to_neo4j,
                parsed,
                doc.page_content,
                getattr(doc, "metadata", {}) or {}
            )

# --- Configuration ---
DOCUMENT_CACHE_PATH = "document_chunks.pkl"
PROGRESS_FILE = "graph_progress.txt"
FAILED_CHUNKS = "failed_chunks.txt"

def sanitize_rel_type(s: str) -> str:
    # Uppercase, replace non [A-Za-z0-9_] with _, and ensure it doesn't start with a digit.
    t = re.sub(r'[^A-Za-z0-9_]', '_', s or '').upper()
    if not t:
        t = "RELATED_TO"
    if t[0].isdigit():
        t = "R_" + t
    return t

def sanitize_label(s: str) -> str:
    # Neo4j labels must be simple tokens; keep your expected ones clean
    t = re.sub(r'[^A-Za-z0-9_]', '_', s or '')
    if not t:
        t = "Entity"
    if t[0].isdigit():
        t = "L_" + t
    return t

# API
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "liquid/lfm2-1.2b")  # or whatever you pick in LM Studio
REQUEST_TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", "120"))  # seconds

# Neo4j Credentials
DATA_DIR = os.path.join(os.getcwd(), "pdfs")
NEO4J_URL = os.getenv("NEO4J_URI")  # "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
print(NEO4J_URL)
print(NEO4J_USER)
print(NEO4J_PASSWORD)

# Model and Processing Configuration
# Switched to llama-cpp: provide your local .gguf via MODEL_PATH env var
# MODEL_PATH = os.getenv("MODEL_PATH")
# if not MODEL_PATH or not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         "MODEL_PATH env var is missing or file not found. "
#         "Run like: MODEL_PATH=/path/to/model.gguf python script.py"
#     )

# Optional performance knobs via env (safe defaults if unset)
N_CTX = int(os.getenv("N_CTX", "8192"))              # context window
N_BATCH = int(os.getenv("N_BATCH", "512"))           # prompt batching inside llama.cpp
N_THREADS = int(os.getenv("N_THREADS", "0"))         # 0 lets library choose
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))   # >0 to offload to GPU via cuBLAS/Metal
MAIN_GPU = int(os.getenv("MAIN_GPU", "0"))

EXTRACTION_BATCH_SIZE = 4  # logical batching for progress/checkpointing

# --- 1. Load Cached Document Chunks ---
if not os.path.exists(DOCUMENT_CACHE_PATH):
    print(f"ERROR: Document cache file not found at '{DOCUMENT_CACHE_PATH}'.")
    print("Please run your initial document loading/splitting script to generate the cache.")
    exit()

print(f"Loading split documents from cache: {DOCUMENT_CACHE_PATH}")
with open(DOCUMENT_CACHE_PATH, "rb") as f:
    documents = pickle.load(f)

print(f"Total document chunks to process: {len(documents)}")

# --- 2. Initialize the Local LLM for Extraction (llama-cpp) ---
# print(f"Initializing llama.cpp model from: {MODEL_PATH}")
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=N_CTX,
#     n_batch=N_BATCH,
#     n_threads=N_THREADS,
#     n_gpu_layers=N_GPU_LAYERS,
#     main_gpu=MAIN_GPU,
#     logits_all=False,
#     verbose=False,
# )
def extract_first_json_object(s: str) -> str | None:
    """
    Return the first top-level JSON object substring from s by tracking braces,
    ignoring braces inside strings/escapes. Returns None if not found.
    """
    in_str = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == '\\':
            if in_str:
                escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return s[start:i+1]
    return None

# --- 3. Define the Extraction Prompt ---
# Keep your original prompt verbatim; we’ll pass it as a plain completion prompt.
extraction_prompt_template = """
You are an expert data analyst specializing in international law and peace agreements. Your task is to extract structured information from the following document chunk.

Analyze the text and extract the following entities and their relationships:
- Agreement: The name or title of the peace agreement.
- Party: Any signatory or participating party (e.g., governments, rebel groups).
- Conflict: The name of the conflict the agreement addresses.
- Location: The city/country where the agreement was signed.
- Topic: Key subjects addressed (e.g., 'ceasefire', 'power-sharing', 'demilitarization').
- Date: The date the agreement was signed.
- Mediator: Any third party that facilitated the agreement.

Based on the text, provide ONLY a JSON object with two keys: "entities" and "relationships".
- "entities" should be a list of objects, each with a "label" (e.g., "Agreement", "Party") and a "name".
- "relationships" should be a list of objects, each with a "source" name, a "target" name, and a "type" (e.g., "SIGNED_BY", "ADDRESSES").

Example Output:
{{
  "entities": [
    {{"label": "Agreement", "name": "The Dayton Accords"}},
    {{"label": "Party", "name": "Republic of Bosnia and Herzegovina"}},
    {{"label": "Topic", "name": "Ceasefire"}}
  ],
  "relationships": [
    {{"source": "The Dayton Accords", "target": "Republic of Bosnia and Herzegovina", "type": "SIGNED_BY"}},
    {{"source": "The Dayton Accords", "target": "Ceasefire", "type": "ADDRESSES"}}
  ]
}}

If no information can be extracted, return an empty JSON object: {{"entities": [], "relationships": []}}.
Do not include any explanation or commentary outside of the JSON object itself.

Document Chunk:
---
{chunk_text}
---

JSON Output:
"""

# --- 4. Neo4j helpers ---

def get_graph_driver():
    """Initializes the Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

def write_to_neo4j(tx, structured_data, chunk_text, source_metadata):
    chunk_id = source_metadata.get('chunk_id') or \
               f"fallback_{source_metadata.get('source','Unknown')}_{source_metadata.get('page',-1)}_{hash(chunk_text)}"

    chunk_node_properties = {
        'text': chunk_text,
        'source': source_metadata.get('source', 'Unknown'),
        'page': source_metadata.get('page', -1),
        'chunkId': chunk_id
    }
    tx.run("MERGE (c:Chunk {chunkId: $props.chunkId}) SET c = $props", props=chunk_node_properties)

    # Entities
    for entity in structured_data.get('entities', []):
        label = sanitize_label(entity.get('label', 'Entity'))
        rawname = entity.get('name', '')
        print("Rawname", rawname)
        name  = entity.get('name', '').strip()
        if not name:
            continue
        tx.run(f"MERGE (n:{label} {{name: $name}})", name=name)
        tx.run(f"""
            MATCH (c:Chunk {{chunkId: $chunk_id}})
            MATCH (e:{label} {{name: $entity_name}})
            MERGE (e)-[:EXTRACTED_FROM]->(c)
        """, chunk_id=chunk_id, entity_name=name)

    # Relationships
    for rel in structured_data.get('relationships', []):
        src = (rel.get('source') or '').strip()
        tgt = (rel.get('target') or '').strip()
        raw_type = rel.get('type') or 'RELATED_TO'
        if not src or not tgt:
            continue
        rel_type = sanitize_rel_type(raw_type)
        # NOTE: dynamic relationship types can’t be parameterized, so we inject the sanitized token
        tx.run(f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            MERGE (source)-[:{rel_type}]->(target)
        """, source_name=src, target_name=tgt)

# --- 5. Progress resume ---

start_chunk = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        try:
            start_chunk = int(f.read()) + 1
            print(f"--- Found progress file. Resuming from chunk {start_chunk}. ---")
        except:
            print("--- Progress file is empty or corrupted. Starting from the beginning. ---")

document_batches = [documents[i:i + EXTRACTION_BATCH_SIZE] for i in range(0, len(documents), EXTRACTION_BATCH_SIZE)]
start_batch = start_chunk // EXTRACTION_BATCH_SIZE
batches_to_process = document_batches[start_batch:]

driver = get_graph_driver()

# --- 6. Inference helper (llama-cpp, per-item) ---

def parse_or_none(output: str):
    js = extract_first_json_object(output)
    if not js:
        return None
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        try:
            fixed = repair_json(js)
            return json.loads(fixed)
        except Exception:
            print("Failed to repair JSON:", js[:200])
            return None



def infer_json_for_chunk(text: str) -> str:
    """
    Calls LM Studio's OpenAI-compatible /v1/chat/completions.
    We push a system message that enforces JSON-only output.
    """
    user_prompt = extraction_prompt_template.format(chunk_text=text) + \
                  "\nReturn only a single minified JSON object. No prose, no backticks."

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": "You output only valid JSON objects. No explanations."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": -1,     # LM Studio accepts -1 for “unlimited”; set a number if your model complains
        "stream": False
    }

    # Try to request structured JSON if the backend supports it
    try:
        payload_with_rf = dict(payload)
        payload_with_rf["response_format"] = {"type": "json_object"}
        res = requests.post(LMSTUDIO_API_URL, json=payload_with_rf, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback without response_format for servers/models that don't support it
        res = requests.post(LMSTUDIO_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


print(f"Starting extraction from batch {start_batch} with concurrency={CONCURRENCY} ...")
for i, batch in enumerate(tqdm(batches_to_process, desc="Extracting Knowledge Graph")):
    current_chunk_index = (start_batch + i) * EXTRACTION_BATCH_SIZE
    try:
        asyncio.run(process_batch_async(batch, driver))
    except RuntimeError:
        # if already in an event loop (e.g., Jupyter), use nest_asyncio or an alternative runner
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_batch_async(batch, driver))
    # decoded_outputs = []
    # # NOTE: llama.cpp does not (yet) do multi-prompt batching via this API, so we iterate.
    # for doc in batch:
    #     try:
    #         decoded_outputs.append(infer_json_for_chunk(doc.page_content))
    #     except Exception as e:
    #         tqdm.write(f"WARNING: Inference failed for a chunk. Error: {e}")
    #         decoded_outputs.append("")

    # for j, output in enumerate(decoded_outputs):
    #     doc_index_in_batch = current_chunk_index + j
    #     if not output:
    #         tqdm.write(f"WARNING: Empty output for chunk {doc_index_in_batch}.")
    #         continue

    #     try:
    #         json_str = extract_first_json_object(output)
    #         if not json_str:
    #             raise AttributeError("No top-level JSON object found in model output.")
    #         data = json.loads(json_str)
    #         parsed = parse_or_none(output)
    #         if parsed is None:
    #             # one targeted retry
    #             retried = infer_json_for_chunk(batch[j].page_content + "\nRemember: Output must be a single compact JSON object.")
    #             parsed = parse_or_none(retried)

    #         if parsed is None:
    #             tqdm.write(f"WARNING: Could not parse JSON for chunk {doc_index_in_batch}.")
    #             continue
    #         with driver.session() as session:
    #             session.execute_write(
    #                 write_to_neo4j,
    #                 parsed,
    #                 batch[j].page_content,
    #                 getattr(batch[j], "metadata", {}) or {}
    #             )
    #     except (json.JSONDecodeError, AttributeError) as e:
    #         tqdm.write(f"WARNING: Could not parse JSON for chunk {doc_index_in_batch}. Error: {e}")
    #         continue

    # --- Checkpoint after each batch ---
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(current_chunk_index + len(batch) - 1))

    # --- Memory cleanup ---
    decoded_outputs = None
    gc.collect()

driver.close()

# If the entire loop completes successfully, remove the progress file.
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

print("\n--- Knowledge Graph Extraction Complete ---")
