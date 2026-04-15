
import os
import pathlib
import json
from dotenv import load_dotenv

from psycopg2.extras import execute_batch, RealDictCursor
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.db import get_db_conn
from src.ingestion.docling_parser import parse_document

load_dotenv(override=True)

# ---------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300

# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# ---------------------------------------------------------------------
# SCHEMA CREATION (🔥 FIX FOR YOUR ERROR)
# ---------------------------------------------------------------------
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT UNIQUE NOT NULL,
    source_path TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,

    content TEXT NOT NULL,

    image_path TEXT,
    mime_type TEXT,

    page_number INT,
    section TEXT,
    source_file TEXT,

    position JSONB,

    embedding VECTOR(1536),

    metadata JSONB
);
"""

# ---------------------------------------------------------------------
# TEXT SPLITTING
# ---------------------------------------------------------------------
def _split_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step

    return chunks


# ---------------------------------------------------------------------
# INGESTION
# ---------------------------------------------------------------------
def run_ingestion(file_path: str, original_filename: str) -> dict:
    pdf_path = pathlib.Path(file_path).resolve()
    print(f"[ingestion] Parsing: {pdf_path}")

    parsed_elements = parse_document(str(pdf_path))
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    chunks = []

    # ---------------- CHUNK PREPARATION ----------------
    for elem in parsed_elements:
        content_type = elem["content_type"]
        content = elem.get("content") or "[Image]"
        metadata = dict(elem.get("metadata", {}))

        page = metadata.get("page_number")
        section = metadata.get("section")

        base_chunk = {
            "chunk_type": content_type,
            "element_type": elem.get("element_type"),
            "image_path": elem.get("image_path"),
            "mime_type": elem.get("mime_type"),
            "page": page,
            "section": section,
            "source": original_filename,
            "position": metadata.get("position"),
        }

        if content_type == "text" and len(content) > _TEXT_CHUNK_SIZE:
            sub_chunks = _split_text(content, _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP)
        else:
            sub_chunks = [content]

        for sub in sub_chunks:
            chunks.append({
                **base_chunk,
                "content": sub,
                "metadata": {
                    "page": page,
                    "section": section,
                    "source": original_filename
                }
            })

    print(f"[ingestion] {len(chunks)} chunks ready")

    # -----------------------------------------------------------------
    # DB OPERATIONS
    # -----------------------------------------------------------------
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # 🔥 Ensure schema exists
        cur.execute(SCHEMA_SQL)
        conn.commit()

        # ---------------- DOCUMENT UPSERT ----------------
        cur.execute(
            """
            INSERT INTO documents (filename, source_path, ingested_at)
            VALUES (%s, %s, now())
            ON CONFLICT (filename)
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                ingested_at = now()
            RETURNING id
            """,
            (original_filename, str(pdf_path))
        )

        doc_id = str(cur.fetchone()["id"])
        print(f"[ingestion] doc_id={doc_id}")

        # ---------------- CHUNKS INSERT ----------------
        insert_sql = """
            INSERT INTO multimodal_chunks (
                doc_id,
                chunk_type,
                element_type,
                content,
                image_path,
                mime_type,
                page_number,
                section,
                source_file,
                position,
                embedding,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        rows = []

        for c in chunks:
            embedding = (
                _embeddings_model.embed_query(c["content"])
                if c["content"]
                else None
            )

            rows.append((
                doc_id,
                c["chunk_type"],
                c["element_type"],
                c["content"],
                c["image_path"],
                c["mime_type"],
                c["page"],
                c["section"],
                c["source"],
                json.dumps(c["position"]) if c["position"] else None,
                embedding,
                json.dumps(c["metadata"]),
            ))

        execute_batch(cur, insert_sql, rows)
        conn.commit()

        print(f"[ingestion] Stored {len(rows)} chunks")

    except Exception as e:
        conn.rollback()
        print("[ingestion ERROR]", e)
        raise

    finally:
        cur.close()
        conn.close()

    return {
        "status": "success",
        "document_id": doc_id,
        "chunks_ingested": len(rows),
    }


# ---------------------------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pdfs = [
        "data\KB_Credit_Card_Spend_Summarizer.pdf"
    ]

    for pdf in pdfs:
        if pathlib.Path(pdf).exists():
            run_ingestion(pdf, os.path.basename(pdf))  # ✅ FIXED
        else:
            print(f"PDF not found: {pdf}")