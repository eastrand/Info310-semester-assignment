import os
import requests
from dotenv import load_dotenv
import torch

from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import MergerRetriever




# ----------------------------
# API (LM Studio)
# ----------------------------
load_dotenv()

LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen3-4b-instruct-2507")  # or whatever you pick in LM Studio
REQUEST_TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", "120"))  # seconds

# Neo4j
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

index_name = "docs_index"
fulltext_index_name = "chunkTextIndex"

# ----------------------------
# Embeddings
# ----------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

embeddings_model = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# ----------------------------
# Retrievers
# ----------------------------
vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings_model,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name=index_name,
)

vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

keyword_retriever = Neo4jVector.from_existing_index(
    embedding=embeddings_model,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name=index_name,
    search_type="hybrid",
    keyword_index_name="chunkTextIndex",
    # search_type="fulltext",
    # search_query=f"CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $query, {{limit:3}}) YIELD node RETURN node",
).as_retriever()

hybrid_retriever = MergerRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.6, 0.4],
)

# ----------------------------
# Prompt
# ----------------------------
template = """
You are an expert assistant on international peace agreements.
Answer the question based only on the following context.
If the context does not contain the answer, state that you don't know.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ----------------------------
# LM Studio call as a Runnable
# ----------------------------
def _lmstudio_call(chat_prompt_value):
    """
    Accepts a ChatPromptValue from LangChain, converts to OpenAI-style messages,
    posts to LM Studio /v1/chat/completions, and returns assistant text.
    """
    # Convert LangChain prompt value to a list of {role, content} messages
    messages = []
    for m in chat_prompt_value.to_messages():
        role = getattr(m, "type", None) or m.__class__.__name__.replace("Message", "").lower()
        # Normalize role names to OpenAI schema
        if role == "human" or role == "user":
            role = "user"
        elif role == "ai" or role == "assistant":
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"
        messages.append({"role": role, "content": m.content})

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": 0.7,
        # Add other parameters you want LM Studio to respect:
        # "top_p": 1.0, "max_tokens": 1024, "presence_penalty": 0, "frequency_penalty": 0,
    }

    try:
        resp = requests.post(
            LMSTUDIO_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except requests.RequestException as e:
        return f"[LM Studio request failed: {e}]"
    except Exception as e:
        return f"[Unexpected error calling LM Studio: {e}]"

lmstudio_runnable = RunnableLambda(_lmstudio_call)


rag_chain = (
    {"context": hybrid_retriever, "question": RunnablePassthrough()}
    | prompt
    | lmstudio_runnable
    | StrOutputParser()
)

if __name__ == "__main__":
    # Example 1: A semantic question where vector search excels
    # question1 = "What are the common themes regarding demilitarization in agreements signed in Africa?"
    # print(f"--- Asking: {question1} ---")
    # response1 = rag_chain.invoke(question1)
    # print(response1)
    # print("\n" + "=" * 50 + "\n")

    # # Example 2: A keyword-heavy question where full-text search excels
    # question2 = "Find the exact text that mentions the 'National Liberation Army' (ELN)."
    # print(f"--- Asking: {question2} ---")
    # response2 = rag_chain.invoke(question2)
    # print(response2)
    test_questions = [
        # ✅ Grounded / real questions
        "Which African peace agreements since 2000 have included explicit provisions for disarmament, demobilization, and reintegration (DDR)?",
        "What mechanisms for power sharing were included in the 2015 South Sudan peace agreement?",
        
        # ⚠️ Hallucination-inducing questions
        "What peace agreement in Antarctica addressed climate refugee relocation in 2018?",
        "Which peace accord in the United States ended the 2011 civil conflict in Texas?",
        "List the provisions on internet governance in the 1998 Good Friday Agreement.",
        "What clauses in the 2004 Darfur peace protocol regulated the use of drones for monitoring?",
        
        ## First examples
        "Find the exact text that mentions the 'National Liberation Army' (ELN).",
        "What are the common themes regarding demilitarization in agreements signed in Africa?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n=== Question {i}: {question} ===")
        try:
            answer = rag_chain.invoke(question)
            print("Answer:\n", answer)
        except Exception as e:
            print("Error during query:", e)
        print("\n" + "-" * 70)