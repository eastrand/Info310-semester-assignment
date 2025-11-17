import os
import pickle
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import log

def load_and_chunk_documents(data_dir="pdfs", chunk_size=1500, chunk_overlap=200, output_file="document_chunks.pkl"):
    """
    Loads documents from a directory (PDFs or .txt), splits them into chunks, and saves as a pickle file.
    """

    all_docs = []
    log(f"📂 Loading documents from: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file in tqdm(files, desc="Loading files"):
            path = os.path.join(root, file)
            try:
                if file.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.lower().endswith(".txt"):
                    loader = TextLoader(path)
                else:
                    continue

                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = file
                all_docs.extend(docs)
            except Exception as e:
                log(f"⚠️ Error loading {file}: {e}")

    if not all_docs:
        log("❌ No documents found.")
        return []

    log(f"✅ Loaded {len(all_docs)} base documents. Now splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    split_docs = splitter.split_documents(all_docs)

    # Assign unique chunk IDs for tracking
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = f"chunk_{i}"

    log(f"🧩 Split into {len(split_docs)} chunks. Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(split_docs, f)

    log("✅ Document chunks saved successfully.")
    return split_docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk and preprocess documents for vector ingestion")
    parser.add_argument("--data-dir", default="pdfs", help="Folder with PDFs or text files")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--output-file", default="document_chunks.pkl")
    args = parser.parse_args()

    load_and_chunk_documents(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_file=args.output_file,
    )
