from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
import torch
from langchain_community.vectorstores import Neo4jVector
from utils import log, detect_device, get_tokenizer_for_embedding

def ingest_vectors(documents, embeddings, args):
    token_counter = get_tokenizer_for_embedding(args)
    total_tokens = 0

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

    if args.reset_index:
        log("Deleting old data from Neo4j…")
        try:
            Neo4jVector.from_existing_index(**vector_store_config).delete(delete_all=True)
            log("Old data deleted.")
        except Exception:
            log("No existing index found to delete. Skipping.")

    try:
        vector_store = Neo4jVector.from_existing_index(**vector_store_config)
        log("Successfully connected to existing index.")
    except Exception:
        log("Index not found. Creating new index.")
        vector_store = Neo4jVector(**vector_store_config)

    document_batches = [documents[i:i + args.batch_size] for i in range(0, len(documents), args.batch_size)]
    log(f"Processing {len(document_batches)} batches with concurrency={args.concurrency} …")

    def process_batch(batch):
        try:
            vector_store.add_documents(
                batch,
                ids=[doc.metadata.get("chunk_id", "") for doc in batch]
            )
        except Exception as e:
            log(f"⚠️ Error processing batch: {e}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(process_batch, batch) for batch in document_batches]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel Embedding Batches"):
            f.result()

    total_tokens = sum(sum(token_counter(doc.page_content) for doc in batch) for batch in document_batches)
    log(f"\n--- Vector Ingest Finished ---")
    log(f"Total Prompt Tokens (Embeddings): {total_tokens}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
