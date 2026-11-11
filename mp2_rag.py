from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is missing from environment variables.")
    sys.exit(1)

client_openai = OpenAI(api_key=OPENAI_API_KEY)

try:
    if QDRANT_URL:
        client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client_qdrant = QdrantClient(path="./qdrant_data")
except Exception as e:
    print(f"Error initializing Qdrant client: {e}")
    sys.exit(1)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory."""
    documents = []
    if not corpus_dir.exists():
        print(f"Warning: Corpus directory '{corpus_dir}' not found.")
        return documents
    
    for file_path in sorted(corpus_dir.glob("*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            title = lines[0] if lines else file_path.name
            documents.append({
                "source": file_path.name,
                "title": title,
                "text": text
            })
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    return documents


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks with section header detection."""
    chunks = []
    text = doc["text"]
    source = doc["source"]
    title = doc["title"]
    
    paragraphs = text.split("\n\n")
    current_section = title
    current_chunk_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(para) < 80 and not para.endswith(('.', '?', '!')):
            current_section = para
            continue

        if len(current_chunk_text) + len(para) <= TARGET_CHUNK_SIZE:
            current_chunk_text += ("\n\n" if current_chunk_text else "") + para
        else:
            if current_chunk_text:
                chunks.append({
                    "source": source,
                    "title": title,
                    "section": current_section,
                    "text": current_chunk_text
                })
            current_chunk_text = para

    if current_chunk_text:
        chunks.append({
            "source": source,
            "title": title,
            "section": current_section,
            "text": current_chunk_text
        })
        
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model."""
    try:
        response = client_openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""
    try:
        client_qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error setting up Qdrant collection: {e}")
        raise


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=chunk
            )
        )
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client_qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + batch_size]
        )
    print(f"Successfully ingested {len(chunks)} chunks into {COLLECTION_NAME}.")


# ─── Sanitizer Guardrail ────────────────────────────────────────────────

def is_valid_query(query: str) -> bool:
    if not query or not isinstance(query, str):
        return False
    cleaned = query.strip()
    if len(cleaned) < 3 or sum(char.isalnum() for char in cleaned) < 2:
        return False
    if not re.search(r'[a-zA-Z]', cleaned):
        return False
    return True


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query with validation guardrails."""
    if not is_valid_query(query):
        print(f"[-] Query rejected by sanitizer: '{query}'")
        return []

    try:
        query_embedding = embed_texts([query])[0]
        results = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=k
        )
        retrieved = []
        for scored_point in results.points:
            payload = scored_point.payload or {}
            payload["score"] = scored_point.score
            retrieved.append(payload)
        return retrieved
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result dictionary."""
    start_time = time.time()
    
    retrieved_chunks = retrieve(question, k=k)
    
    citations = []
    context_blocks = []
    for chunk in retrieved_chunks:
        source_label = f"[Source: {chunk.get('title', 'Unknown')} — {chunk.get('section', 'General')}]"
        context_blocks.append(f"{source_label}\n{chunk.get('text', '')}")
        citations.append({
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", "")
        })

    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    user_message = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"

    try:
        response = client_openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        ans_text = response.choices[0].message.content.strip()
    except Exception as e:
        ans_text = f"Error generating answer from LLM: {e}"

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "question": question,
        "answer": ans_text,
        "citations": citations,
        "latency_ms": latency_ms
    }


# ─── Output Logger Class (Writes to Console + File simultaneously) ───────

class DualOutputLogger:
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ─── Validation harness ──────────────────────────────────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        
        # Handle expected_source as either a string or a list of acceptable sources
        expected = q["expected_source"]
        if isinstance(expected, list):
            source_hit = any(src in cited_sources for src in expected)
            expected_str = " | ".join(expected)
        else:
            source_hit = expected in cited_sources
            expected_str = expected

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"     Q: {q['question']}")
        print(f"     Answer:\n     {result['answer']}")
        print(f"     Cited: {', '.join(cited_sources)}")
        print(f"     Expected: {expected_str}")
        print(f"     Facts matched: {facts_hit}/{facts_total}")
        print(f"     Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI ─────────────────────────────────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    output_file = Path(__file__).parent / "mp2_validation.txt"
    logger = DualOutputLogger(output_file)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        validate_against(DATA_DIR / "predefined_questions.jsonl")
        learner_path = DATA_DIR / "learner_questions.jsonl"
        if learner_path.exists():
            lines = [l for l in learner_path.read_text().splitlines() if l.strip()]
            if lines:
                first = json.loads(lines[0])
                if not first.get("question", "").startswith("Replace this"):
                    validate_against(learner_path)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Validation report successfully created at: {output_file.resolve()}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            client_qdrant.close()
        except Exception:
            pass