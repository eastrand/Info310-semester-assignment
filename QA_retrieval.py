import os
import json
import sqlite3
from datetime import datetime
from typing import List

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SQLite logging
# ============================================================

QA_DB_PATH = "qa_answers.db"


def init_qa_db():
    conn = sqlite3.connect(QA_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            fusion_enabled INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_qa_answer(question: str, answer: str, fusion_enabled: bool):
    conn = sqlite3.connect(QA_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO answers (timestamp, question, answer, fusion_enabled)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        question,
        answer,
        1 if fusion_enabled else 0
    ))
    conn.commit()
    conn.close()


# ============================================================
# Neo4j vector search helper
# ============================================================

def neo4j_query(driver, query: str, params: dict):
    with driver.session() as session:
        return session.run(query, params).data()


# ============================================================
# RAG fusion scoring (optional)
# ============================================================

def reciprocal_rank_fusion(results: List[List[dict]], k=60):
    """
    RRF scoring to merge vector + keyword + KG results
    """
    scores = {}

    for result_list in results:
        for rank, item in enumerate(result_list):
            doc_id = item["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in ranked]


# ============================================================
# Answer synthesis using LLM
# ============================================================

def generate_answer(context: str, question: str, client: OpenAI):
    prompt = f"""
    You are an expert analyst. Using the retrieved excerpts below, extract and list **the peace agreements** that contain explicit DDR (Disarmament, Demobilization, Reintegration) provisions.

    You MUST:
    - Identify the agreement names
    - Identify the year
    - Only cite agreements actually mentioned in the context
    - Synthesize across multiple chunks when needed
    - Never answer "not stated" if the information can be inferred from the context

    === RETRIEVED CONTEXT ===
    {context}

    === QUESTION ===
    {question}

    Provide a concise, bullet-point list of agreements.
    """

    response = client.chat.completions.create(
        # model=os.getenv("LLM_MODEL", "gpt-5-mini"),
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
        # temperature=0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Main QA function
# ============================================================

def answer_question(question: str, use_fusion: bool = False):
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

    # OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------
    # 1. Vector search
    # ------------------------------------------------------------
    vec_query = """
    CALL db.index.vector.queryNodes('chunk_embeddings', 5, $embedding)
    YIELD node, score
    RETURN node.doc_id AS doc_id, node.text AS text, score
    """

    # Embed question
    embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
    ).data[0].embedding

    vec_results = neo4j_query(driver, vec_query, {"embedding": embedding})

    # ------------------------------------------------------------
    # 2. Keyword search (Cypher fulltext)
    # ------------------------------------------------------------
    kw_query = """
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $q)
    YIELD node, score
    RETURN node.doc_id AS doc_id, node.text AS text, score
    LIMIT 5
    """

    kw_results = neo4j_query(driver, kw_query, {"q": question})

    # ------------------------------------------------------------
    # 3. KG hop search (based on keywords)
    # ------------------------------------------------------------
    kg_query = """
    MATCH (e)-[:DERIVED_FROM]->(d:Document)
    WHERE e.name CONTAINS $q
    RETURN d.doc_id AS doc_id, d.text AS text, 1 AS score
    LIMIT 5
    """

    kg_results = neo4j_query(driver, kg_query, {"q": question})

    # ------------------------------------------------------------
    # 4. Fusion or vanilla RAG
    # ------------------------------------------------------------
    if use_fusion:
        doc_ids = reciprocal_rank_fusion([vec_results, kw_results, kg_results])
        top_contexts = []
        for d in doc_ids[:4]:
            match = next((x for x in vec_results + kw_results + kg_results if x["doc_id"] == d), None)
            if match:
                top_contexts.append(match["text"])
    else:
        # Just vector search
        top_contexts = [r["text"] for r in vec_results[:4]]

    context = "\n\n".join(top_contexts)
## Debug
    print("\n---- VECTOR RESULTS ----")
    print(vec_results)

    print("\n---- FULLTEXT RESULTS ----")
    print(kw_results)

    print("\n---- CONTEXT SENT TO MODEL ----")
    print(context[:2000])
    print("... (truncated) ...")
    # ------------------------------------------------------------
    # 5. Generate LLM answer
    # ------------------------------------------------------------
    answer = generate_answer(context, question, client)

    return answer


# ============================================================
# CLI ENTRYPOINT — with SQLite logging
# ============================================================

if __name__ == "__main__":
    import argparse
    import sys

    init_qa_db()

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", help="Single question")
    parser.add_argument("--questions-file", help="File containing one question per line")
    parser.add_argument("--fusion", action="store_true", help="Enable RAG Fusion scoring")
    args = parser.parse_args()

    questions = []

    if args.q:
        questions.append(args.q)

    if args.questions_file and os.path.exists(args.questions_file):
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions.extend([line.strip() for line in f.readlines() if line.strip()])

    if not questions:
        print("No questions provided. Nothing to do.")
        sys.exit(0)

    for q in questions:
        print("\n====================================================")
        print(f"QUESTION: {q}")
        print("====================================================\n")

        ans = answer_question(q, use_fusion=args.fusion)

        print("ANSWER:\n")
        print(ans)
        print("\n----------------------------------------------------\n")

        # Log into DB
        log_qa_answer(q, ans, args.fusion)
