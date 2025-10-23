import os
import re
import pickle
import gc
import json
from langchain_core.documents import Document
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j import GraphDatabase

# --- Configuration ---
# File paths
DOCUMENT_CACHE_PATH = "document_chunks.pkl"
PROGRESS_FILE = "graph_progress.txt"

# Neo4j Credentials
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test1234" # Change if needed

# Model and Processing Configuration
# Phi-3-mini is an excellent choice for balancing speed and quality.
# It's small, fast, and very capable at following instructions.
EXTRACTION_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
EXTRACTION_BATCH_SIZE = 4  # Keep this low for LLM inference to conserve VRAM

# --- 1. Load Cached Document Chunks ---
# The script assumes the 'document_chunks.pkl' file already exists.
# If not, you'll need to run your initial document processing script first.
if not os.path.exists(DOCUMENT_CACHE_PATH):
    print(f"ERROR: Document cache file not found at '{DOCUMENT_CACHE_PATH}'.")
    print("Please run your initial document loading/splitting script to generate the cache.")
    exit()

print(f"Loading split documents from cache: {DOCUMENT_CACHE_PATH}")
with open(DOCUMENT_CACHE_PATH, "rb") as f:
    documents = pickle.load(f)

print(f"Total document chunks to process: {len(documents)}")

# --- 2. Initialize the Local LLM for Extraction ---
print(f"Initializing extraction model: {EXTRACTION_MODEL_ID}")

# Define quantization configuration for memory efficiency
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(EXTRACTION_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    EXTRACTION_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True # Required by some models like Phi-3
)

# --- 3. Define the Extraction Prompt ---
# A clear, structured prompt is critical for getting reliable JSON output.
# Curly braces in the example JSON are "escaped" by doubling them up (e.g., {{, }})
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

# --- 4. Main Extraction and Graph Building Loop ---

def get_graph_driver():
    """Initializes the Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

def write_to_neo4j(tx, structured_data, chunk_text, source_metadata):
    """
    A transactional function to write extracted nodes and relationships to Neo4j.
    It also creates a :Chunk node for the original text.
    """
    # Create a node for the original text chunk for context
    chunk_node_properties = {
        'text': chunk_text,
        'source': source_metadata.get('source', 'Unknown'),
        'page': source_metadata.get('page', -1)
    }
    tx.run("CREATE (c:Chunk $props)", props=chunk_node_properties)
    
    # Create nodes for all extracted entities, using MERGE to avoid duplicates
    for entity in structured_data.get('entities', []):
        tx.run(f"MERGE (n:{entity['label']} {{name: $name}})", name=entity['name'])

    # Create relationships between the entities
    for rel in structured_data.get('relationships', []):
        tx.run(f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            MERGE (source)-[:{rel['type']}]->(target)
        """, source_name=rel['source'], target_name=rel['target'])

# --- Auto-resume logic: Check for a progress file ---
start_chunk = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        try:
            # Read the number of the last successfully processed chunk
            start_chunk = int(f.read()) + 1
            print(f"--- Found progress file. Resuming from chunk {start_chunk}. ---")
        except:
            print("--- Progress file is empty or corrupted. Starting from the beginning. ---")

# Create batches of documents to process
document_batches = [documents[i:i + EXTRACTION_BATCH_SIZE] for i in range(0, len(documents), EXTRACTION_BATCH_SIZE)]
# Calculate the correct starting batch based on the progress file
start_batch = start_chunk // EXTRACTION_BATCH_SIZE
batches_to_process = document_batches[start_batch:]

driver = get_graph_driver()

print(f"Starting extraction from batch {start_batch}...")
for i, batch in enumerate(tqdm(batches_to_process, desc="Extracting Knowledge Graph")):
    current_chunk_index = (start_batch + i) * EXTRACTION_BATCH_SIZE
    
    # Create the prompts for the current batch of documents
    prompts = [extraction_prompt_template.format(chunk_text=doc.page_content) for doc in batch]
    
    # Use the model's chat template for better instruction following
    messages = [[{"role": "user", "content": p}] for p in prompts]
    
    # Tokenize and generate responses from the LLM
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True, add_generation_prompt=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
    
    for j, output in enumerate(decoded_outputs):
        doc_index_in_batch = current_chunk_index + j
        
        # Robustly extract the JSON part of the LLM's response
        try:
            # Find the first '{' and the last '}' to get the JSON block
            json_str = re.search(r'\{.*\}', output, re.DOTALL).group(0)
            data = json.loads(json_str)
            
            # Write the extracted data to Neo4j in a single transaction
            with driver.session() as session:
                session.write_transaction(
                    write_to_neo4j, 
                    data, 
                    batch[j].page_content, 
                    batch[j].metadata
                )
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, log a warning and continue
            tqdm.write(f"WARNING: Could not parse JSON for chunk {doc_index_in_batch}. Error: {e}")
            continue
    
    # --- Checkpointing: Update progress file after each successful batch ---
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(current_chunk_index + len(batch) - 1))

    # --- Aggressive Memory Cleanup: Prevents memory leaks on long runs ---
    gc.collect()
    torch.cuda.empty_cache()

driver.close()

# If the entire loop completes successfully, remove the progress file.
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

print("\n--- Knowledge Graph Extraction Complete ---")

```

### **Important: Installation & Setup**

This script has specific dependencies. Please follow these steps in a **new, clean virtual environment** to avoid conflicts.

**Step 1: Create a Clean Virtual Environment**

```bash
python -m venv kg_venv
```

**Step 2: Activate the Environment**

```powershell
.\kg_venv\Scripts\Activate.ps1
```
Your terminal prompt should now show `(kg_venv)`.

**Step 3: Install Required Libraries**

This single command installs all the necessary packages for running the local LLM with GPU acceleration.

```bash
pip install torch transformers accelerate bitsandbytes neo4j langchain-core tqdm
```

**Step 4: Run the Script**

Once the installation is complete, you can run the script. It will automatically download the `microsoft/Phi-3-mini-4k-instruct` model on the first run.

```bash
python build_knowledge_graph.py
