import os
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import torch

# --- 1. Load Configuration ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
index_name = "docs_index"
fulltext_index_name = "chunkTextIndex"

# --- 2. Initialize Models ---
# Initialize the local LLM you want to use for answering questions
llm = ChatOllama(model="llama3")

# Initialize the same embedding model used for ingestion
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

embeddings_model = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# --- 3. Set Up the Hybrid Retriever ---

# Initialize the vector store object to connect to your existing index
vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings_model,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name=index_name,
)

# K-Nearest Neighbors (KNN) or vector search retriever
vector_retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# Keyword or full-text search retriever
# This uses the full-text index we created
keyword_retriever = Neo4jVector.from_existing_index(
    embedding=embeddings_model,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name=index_name,
    search_type="fulltext",
    search_query=f"CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $query, {{limit:3}}) YIELD node RETURN node"
).as_retriever()


# The EnsembleRetriever combines the results of both
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.6, 0.4] # Give slightly more weight to the semantic vector search
)

# --- 4. Create the RAG Chain ---
template = """
You are an expert assistant on international peace agreements.
Answer the question based only on the following context.
If the context does not contain the answer, state that you don't know.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the RAG pipeline
rag_chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Ask Questions ---
if __name__ == "__main__":
    # Example 1: A semantic question where vector search excels
    question1 = "What are the common themes regarding demilitarization in agreements signed in Africa?"
    print(f"--- Asking: {question1} ---")
    response1 = rag_chain.invoke(question1)
    print(response1)
    print("\n" + "="*50 + "\n")

    # Example 2: A keyword-heavy question where full-text search excels
    question2 = "Find the exact text that mentions the 'National Liberation Army' (ELN)."
    print(f"--- Asking: {question2} ---")
    response2 = rag_chain.invoke(question2)
    print(response2)