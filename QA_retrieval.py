import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# LLM RAW RESPONSE LOGGING DB
# ------------------------------------------------------------

conn = sqlite3.connect("qa_logs.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS llm_raw_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    question TEXT,
    prompt TEXT,
    raw_response TEXT
)
""")

conn.commit()
conn.close()

# ------------------------------------------------------------
# Retrieval / context knobs
# ------------------------------------------------------------

TOP_K_VECTOR = 25          # how many vector hits to pull
TOP_K_KEYWORD = 25         # how many fulltext hits to pull (when fusion is on)
TOP_K_KG = 15              # how many KG hits to pull (when fusion is on)
TOP_K_FUSED = 20           # how many fused results to keep for context

MAX_CONTEXT_CHARS = 24000  # hard cap on context size passed to GPT
MAX_CHARS_PER_CHUNK = 2000 # truncate very long chunks for safety

# ============================================================
# SQLite logging for final QA answers
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
# Neo4j helper
# ============================================================

def neo4j_query(driver, query: str, params: dict) -> List[Dict]:
    with driver.session() as session:
        return session.run(query, params).data()


# ============================================================
# Retrieval helpers (vector, fulltext, KG)
# ============================================================

def get_vector_results(question: str, driver, client: OpenAI) -> List[Dict]:
    """Retrieve top chunks by vector similarity."""
    embedding = client.embeddings.create(
        model="text-embedding-3-small",  # must match ingest
        input=question
    ).data[0].embedding

    query = f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', $embedding, {TOP_K_VECTOR})
    YIELD node, score
    RETURN node.doc_id AS doc_id, node.text AS text, score
    ORDER BY score DESC
    """

    return neo4j_query(driver, query, {"embedding": embedding})


def get_fulltext_results(question: str, driver) -> List[Dict]:
    """Retrieve top chunks using fulltext index on :Chunk(text)."""
    query = f"""
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $q)
    YIELD node, score
    RETURN node.doc_id AS doc_id, node.text AS text, score
    ORDER BY score DESC
    LIMIT {TOP_K_KEYWORD}
    """
    return neo4j_query(driver, query, {"q": question})


def get_kg_results(question: str, driver) -> List[Dict]:
    """
    Simple KG-based retrieval:
    find Documents linked from entities whose name contains the query string.
    """
    query = f"""
    MATCH (e)-[:DERIVED_FROM]->(d:Document)
    WHERE e.name CONTAINS $q
    RETURN d.doc_id AS doc_id, d.text AS text, 1.0 AS score
    LIMIT {TOP_K_KG}
    """
    return neo4j_query(driver, query, {"q": question})


# ============================================================
# RAG fusion scoring (chunk-level)
# ============================================================

def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = TOP_K_FUSED) -> List[Dict]:
    """
    Reciprocal Rank Fusion over chunk-level results.
    Each result is a dict with at least: doc_id, text, score.
    We dedupe by (doc_id, text) so multiple chunks from the same doc
    can still appear, but identical duplicates won't.
    """
    def _index_map(results: List[Dict]):
        return {(r["doc_id"], r["text"]): idx for idx, r in enumerate(results)}

    # Build index maps
    maps = [_index_map(results) for results in result_lists]

    # All unique (doc_id, text) keys
    all_keys = set()
    for m in maps:
        all_keys |= set(m.keys())

    scores = {}
    for key in all_keys:
        rr = 0.0
        for m in maps:
            if key in m:
                # standard RRF scoring
                rr += 1.0 / (60 + m[key])
        scores[key] = rr

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Turn back into list of result dicts
    fused_results = []
    # Flatten all results for lookup
    flat = []
    for lst in result_lists:
        flat.extend(lst)

    for (doc_id, text), score in ranked[:k]:
        # find one matching entry to copy fields from
        match = next((r for r in flat if r["doc_id"] == doc_id and r["text"] == text), None)
        if match:
            fused_results.append({
                "doc_id": doc_id,
                "text": text,
                "score": score
            })

    return fused_results


# ============================================================
# Context building
# ============================================================

def build_context(results: List[Dict]) -> str:
    """
    Build a structured context string from retrieval results, with:
    - doc headers
    - chunk truncation
    - global char budget
    """
    pieces = []
    total_chars = 0

    for r in results:
        doc_id = r.get("doc_id", "unknown_doc")
        text = r.get("text", "") or ""

        if not text:
            continue

        # Truncate overly long chunks for safety
        if len(text) > MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK] + " ... [truncated]"

        section = f"[DOC: {doc_id}]\n{text}"

        # Enforce global context budget
        if total_chars + len(section) + 5 > MAX_CONTEXT_CHARS:
            break

        pieces.append(section)
        total_chars += len(section) + 5  # +5 for separators

    return "\n\n-----\n\n".join(pieces)


# ============================================================
# Answer synthesis using LLM
# ============================================================

def generate_answer(context: str, question: str, client: OpenAI):
    prompt = f"""
You are an expert analyst answering questions about peace agreements, security arrangements, and related political texts.

You MUST:
- Use ONLY the information in the context below (no outside knowledge).
- Be as EXHAUSTIVE as possible given the context.
- Synthesize across multiple documents and chunks when needed.
- If the context gives only a partial answer, say clearly: "This answer is based only on the retrieved documents and may be incomplete."

CONTEXT:
{context}

QUESTION:
{question}

Answer in clear, structured bullet points (or short paragraphs if more natural).
"""

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=10000,
        # temperature=0,  # gpt-5 doesn't support custom temperature yet
    )

    raw_json = json.dumps(response.model_dump(), indent=2)

    # Store in SQLite (raw LLM response)
    conn = sqlite3.connect("qa_logs.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO llm_raw_responses (timestamp, question, prompt, raw_response)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        question,
        prompt,
        raw_json
    ))
    conn.commit()
    conn.close()

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
    # 1. Retrieval
    # ------------------------------------------------------------
    vec_results = get_vector_results(question, driver, client)
    kw_results = get_fulltext_results(question, driver)
    kg_results = get_kg_results(question, driver)

    # ------------------------------------------------------------
    # 2. Fusion or vanilla RAG
    # ------------------------------------------------------------
    if use_fusion:
        fused = reciprocal_rank_fusion([vec_results, kw_results, kg_results])
        retrieval_used = fused
    else:
        # Just vector search; keep top-K_FUSED
        retrieval_used = vec_results[:TOP_K_FUSED]

    # ------------------------------------------------------------
    # 3. Build context string
    # ------------------------------------------------------------
    context = build_context(retrieval_used)

    # (Optional debug)
    # print("\n---- VECTOR RESULTS ----")
    # print(vec_results)
    # print("\n---- FULLTEXT RESULTS ----")
    # print(kw_results)
    # print("\n---- KG RESULTS ----")
    # print(kg_results)
    # print("\n---- CONTEXT SENT TO MODEL (truncated) ----")
    # print(context[:2000])
    # print("... (truncated) ...")

    # If somehow nothing was retrieved
    if not context.strip():
        answer = (
            "I could not retrieve any relevant context from the database for this question, "
            "so I cannot provide a grounded answer."
        )
        return answer

    # ------------------------------------------------------------
    # 4. Generate LLM answer
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
