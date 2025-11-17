import pickle
import json
import os
from pathlib import Path
from tqdm import tqdm
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
import torch
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


def ingest_vectors(pkl_path, embeddings, args):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} documents for vector ingestion")
    
    # Setup progress tracking
    progress_file = Path(args.progress_file)
    completed_docs = set()
    
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
            completed_docs = set(progress_data.get("completed_docs", []))
        print(f"Resuming: {len(completed_docs)} documents already processed")
    
    # Filter out already completed documents
    docs_to_process = [d for d in dataset if d["doc_id"] not in completed_docs]
    print(f"Processing {len(docs_to_process)} remaining documents")
    
    vector_store_config = {
        "embedding": embeddings,
        "url": args.neo4j_url,
        "username": args.neo4j_user,
        "password": args.neo4j_password,
        "index_name": args.index_name,
        "node_label": "Chunk",
        "text_node_property": "text",
        "embedding_node_property": "embedding"
    }
    
    # Create vector store (index auto-created if needed)
    vector_store = Neo4jVector(**vector_store_config)
    
    def process_doc(doc_entry):
        doc_id = doc_entry["doc_id"]
        chunks = doc_entry["chunks"]
        
        # Store main document node
        query_doc = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.source = $source, d.text = $text
        """
        vector_store._driver.execute_query(
            query_doc, 
            doc_id=doc_id, 
            source=chunks[0].metadata["source"], 
            text=doc_entry["document_text"]
        )
        
        # Embed and store chunks
        vector_store.add_documents(
            chunks, 
            ids=[c.metadata["chunk_id"] for c in chunks]
        )
        
        # Link document to chunks
        link_query = """
        MATCH (d:Document {doc_id: $doc_id})
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {id: cid})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        vector_store._driver.execute_query(
            link_query, 
            doc_id=doc_id, 
            chunk_ids=[c.metadata["chunk_id"] for c in chunks]
        )
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return doc_id
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_doc, d): d for d in docs_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing docs"):
            try:
                doc_id = future.result()
                completed_docs.add(doc_id)
                
                # Save progress periodically
                with open(progress_file, "w") as f:
                    json.dump({"completed_docs": list(completed_docs)}, f)
            except Exception as e:
                doc_entry = futures[future]
                print(f"⚠️ Error processing {doc_entry['doc_id']}: {e}")
    
    print("✅ Vector ingestion complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="document_chunks.pkl")
    parser.add_argument("--neo4j-url", default=os.getenv("NEO4J_URL"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--index-name", default="peace_index")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--progress-file", default="vector_ingestion_progress.json")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    args = parser.parse_args()
    
    # Initialize embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model=args.embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    
    ingest_vectors(args.pkl, embeddings, args)