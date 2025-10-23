import os
import re
import pickle
import gc  # Import the Garbage Collector module
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DATA_DIR = os.path.join(os.getcwd(), "pdfs")
NEO4J_URL = os.getenv("NEO4J_URI")#"bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
BATCH_SIZE = 16 # Using a smaller batch size for more frequent memory cleanup
DOCUMENT_CACHE_PATH = "document_chunks.pkl"

# --- 1. Load and Split Documents (with Caching) ---
if os.path.exists(DOCUMENT_CACHE_PATH):
    print(f"Loading split documents from cache: {DOCUMENT_CACHE_PATH}")
    with open(DOCUMENT_CACHE_PATH, "rb") as f:
        documents = pickle.load(f)
else:
    print("Loading and splitting documents from source...")
    pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_documents = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = splitter.split_documents(raw_documents)
    
    print("Assigning unique IDs to each document chunk...")
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        # Create a simple, human-readable ID
        doc.metadata["chunk_id"] = f"{os.path.basename(source)}-page:{page}-chunk:{i}"

    print(f"Saving {len(documents)} split documents (with IDs) to cache...")
    with open(DOCUMENT_CACHE_PATH, "wb") as f:
        pickle.dump(documents, f)

print(f"Total document chunks to process: {len(documents)}")

# --- 2. Initialize Embeddings Model ---
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")
if device == 'mps':
    model_name = "papr-ai/Qwen3-Embedding-4B-CoreML"
else: 
    model_name = "Qwen/Qwen3-Embedding-0.6B"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"Initialized embedding model: {model_name}")

# --- 3. Store Embeddings in Neo4j (with Explicit Memory Management) ---

# --- Set to 0 for a fresh run. Set to the batch number that failed to resume. ---
START_FROM_BATCH = 0

# On a fresh run (START_FROM_BATCH = 0), this block should be active.
# When resuming, comment it out to preserve your progress.
if START_FROM_BATCH == 0:
    print("Deleting old data from Neo4j...")
    try:
        vector_store_to_clear = Neo4jVector.from_existing_index(
            embedding=embeddings_model, url=NEO4J_URL, username=NEO4J_USER,
            password=NEO4J_PASSWORD, index_name="docs_index"
        )
        vector_store_to_clear.delete(delete_all=True)
        print("Old data deleted.")
    except Exception:
        print("No existing index found to delete. Skipping.")

# Create all batches from the full documents list
document_batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]

# Determine which batches to process for this run
batches_to_process = document_batches[START_FROM_BATCH:]

if batches_to_process:
    vector_store = None
    if START_FROM_BATCH == 0:
        print("Creating index with the first batch of documents...")
        first_batch = batches_to_process[0]
        vector_store = Neo4jVector.from_documents(
            first_batch, embedding=embeddings_model, url=NEO4J_URL, username=NEO4J_USER,
            password=NEO4J_PASSWORD, index_name="docs_index", node_label="Chunk",
            text_node_property="text", embedding_node_property="embedding"
        )
        print("Index created and first batch added.")
        batches_to_process = batches_to_process[1:]
    else:
        print(f"Connecting to existing index to resume from batch {START_FROM_BATCH}...")
        vector_store = Neo4jVector.from_existing_index(
            embedding=embeddings_model, url=NEO4J_URL, username=NEO4J_USER,
            password=NEO4J_PASSWORD, index_name="docs_index"
        )
        print("Successfully connected to existing index.")

    if batches_to_process and vector_store:
        print(f"Processing {len(batches_to_process)} remaining batches...")
        for batch in tqdm(batches_to_process, desc="Embedding and Storing Batches"):
            vector_store.add_documents(batch, ids=[doc.metadata.get("source", "") + str(i) for i, doc in enumerate(batch)])
            
            # --- EXPLICIT MEMORY CLEANUP ---
            # This is the critical fix. After each batch is processed and stored,
            # we force Python's garbage collector to run and tell PyTorch to
            # empty its cache of unused memory from the GPU. This prevents the
            # memory leak you correctly identified.
            gc.collect()
            torch.cuda.empty_cache()

print("\n--- Script Finished ---")
print("All documents have been embedded and stored in Neo4j.")

