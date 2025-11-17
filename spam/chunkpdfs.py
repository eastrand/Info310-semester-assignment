import os
import pickle
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_pdfs(data_dir: str, chunk_size: int = 1500, chunk_overlap: int = 200, output_path="document_chunks.pkl"):
    """
    Loads and chunks PDF and TXT files from a directory.
    - PDF: uses PyPDFLoader to extract text from pages
    - TXT: reads directly from file
    Outputs a pickle file with each document’s full text and its chunks.
    """
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for filename in tqdm(os.listdir(data_dir), desc="Processing files"):
        file_path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename.lower())[1]

        if ext not in [".pdf", ".txt"]:
            continue

        try:
            # Extract full text
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

            doc_id = filename

            # Split into chunks
            chunks = text_splitter.split_text(full_text)
            chunk_docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk_id": f"{doc_id}-chunk:{i}",
                        "doc_id": doc_id,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            all_docs.append({
                "doc_id": doc_id,
                "document_text": full_text,
                "chunks": chunk_docs,
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    with open(output_path, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"✅ Saved {len(all_docs)} documents (PDF+TXT) and their chunks to {output_path}")
    return all_docs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--output", default="document_chunks.pkl")
    args = parser.parse_args()

    chunk_pdfs(args.data_dir, args.chunk_size, args.chunk_overlap, args.output)
