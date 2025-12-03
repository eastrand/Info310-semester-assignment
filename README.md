# Peace Agreement Pipeline

This jupyter file contains a complete pipeline for collecting, processing, and analyzing peace agreement documents. The code downloads PDFs, does document processing and chunking, vector store ingestion, and question answering.

# Overview

The main steps covered by this project are:

1. Scraping PDF Links
   Use Selenium to browse and collect agreement PDF links from a specified website.

2. Multithreaded PDF Download  
   Download all collected PDFs efficiently with multithreading.

3. Document Chunking  
   Split PDF and text files into manageable chunks for downstream processing.

4. Vector Store Ingestion  
   Store the chunked data in a Neo4j vector database for semantic search.

5. QA Retrieval and Evaluation  
   Question answering pipeline powered by OpenAI for retrieving context and generating responses. Evaluation scripts are provided to assess retrieval quality.

# Setup

1.Install Dependencies

Make sure you have Python 3.9+ installed.  
Recommended: Use a virtual environment.

pip install -r requirements.txt

Some dependencies may need to be installed manually:

pip install:

- selenium webdriver_manager 
   
- requests 
   
- langchain-community 

- langchain-core 

- langchain-openai 
   
- deepeval 
   
- tqdm 
   
- python-dotenv 
   
- openai 
   
- neo4j

2. Environment Configuration

Set up environment variables for Neo4j and OpenAI in a .env file:

NEO4J_URL=bolt://localhost:7687

NEO4J_USER=neo4j

NEO4J_PASSWORD=your_password

OPENAI_API_KEY=your_openai_api_key

# Usage

1. Collect PDF Links  
   Run the PDF finder section or script to fetch links.

2. Download PDFs  
   Use the downloader to save the files locally.

3. Chunk Documents  
   Organize the PDFs and any text files, then run the chunking script to generate processed chunks.

4. Ingest Data  
   Run the ingestion script to add documents/chunks to Neo4j.

5. QA Pipeline  
   Ask questions via the QA retrieval script (supports fusion and standard vector search).

# Evaluation

The repository includes tools for evaluating retrieval and answer quality using DeepEval metrics.  

# Notes

- Make sure your Neo4j database is running and accessible before ingestion or QA.
