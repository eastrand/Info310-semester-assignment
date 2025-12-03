README – Instructions for running the pipeline in Jupyter

This project processes the Peace Agreements dataset, downloads PDFs, performs OCR when needed, chunks documents, ingests them into Neo4j with vector embeddings, extracts a knowledge graph, and provides a RAG-based QA interface. This document explains how to run the pipeline assuming you only have normal Python installed.

NOTE: there might be some redundant or unused pip installs in the install section

Install Jupyter Notebook
Run:
pip install notebook

Start Jupyter:
jupyter notebook

Install Tesseract OCR (optional but recommended)
This is required for converting image-based PDFs into text. If you skip it, the OCR step in the notebook will skip automatically, but the final dataset will be missing around 300 PDFs.

macOS:
brew install tesseract

Ubuntu/Debian:
sudo apt install tesseract-ocr

Windows:
Download from:
https://github.com/UB-Mannheim/tesseract/wiki

The rest of the pipeline still works without OCR.

Create a .env file
Create a file named .env in the project folder with the following values:

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_key_here
LLM_BASE_URL=https://api.openai.com/v1

Start your Neo4j instance before running ingestion or KG extraction.

Running the pipeline
Open the notebook in Jupyter and run the cells from top to bottom in order.

a) PDF Finder
Automatically scrapes the Peace Agreements website and collects all available PDF URLs.

b) PDF Downloader
Downloads all PDFs into the ./pdfs/ directory.

c) Detect Image-Based PDFs
Classifies which PDFs contain text vs. which are scanned and require OCR.

d) OCR (optional)
If Tesseract is installed, OCR is performed and text files are saved to ./txts/.
If not installed, this step is skipped and you will simply have fewer documents.

e) Chunking
Splits documents (PDFs and OCR text) into overlapping text chunks and creates document_chunks.pkl.

f) Vector Ingestion
Loads documents and chunks into Neo4j, applies embeddings, and stores vector representations.

g) Knowledge Graph Extraction
The LLM extracts entities and relationships from documents and inserts them into Neo4j as a structured knowledge graph.

h) RAG QA Retrieval
Allows querying the system using vector search, full-text search, KG search, and fusion scoring.

i) Evaluation
Runs DeepEval metrics if you provide a CSV of evaluation questions.

Outputs
pdfs/ → downloaded PDFs
txts/ → OCR output text files (only if Tesseract is installed)
discarded/ → image PDFs that have been OCR-processed
document_chunks.pkl → chunked text dataset
Neo4j database → contains Documents, Chunks, Entities, and Relationships
qa_logs.db, kg_raw_responses.db, qa_answers.db → logs and evaluation data

Notes
You can run the whole pipeline except OCR without installing anything except Jupyter and Python libraries.
Skipping OCR results in a smaller dataset but does not break the rest of the pipeline.
