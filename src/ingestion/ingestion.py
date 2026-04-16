import os
import json
import pathlib
from typing import List, Dict

from dotenv import load_dotenv
from psycopg2.extras import execute_batch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.db import get_db_conn
from src.ingestion.docling_parser import parse_document

load_dotenv(override=True)

TEXT_CHUNK_SIZE = 1500
TEXT_CHUNK_OVERLAP = 300


_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDINGS_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        chunks.append(text[start:start + size])
        start += step

    return chunks


def run_ingestion(file_path: str) -> dict:
    pdf_path = pathlib.Path(file_path).resolve()
    print(f"[ingestion] Parsing: {pdf_path}")

    parsed_elements: List[Dict] = parse_document(str(pdf_path))
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    chunks: List[Dict] = []

    for elem in parsed_elements:
        content_type = elem.get("content_type")
        metadata = elem.get("metadata", {})

        if content_type == "text":
            content = elem.get("content", "")
            if not content.strip():
                continue

            if len(content) > TEXT_CHUNK_SIZE:
                for sub in _split_text(content, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP):
                    chunks.append({
                        "chunk_type": "text",
                        "content": sub,
                        "image_path": None,
                        "page_number": metadata.get("page_number"),
                        "section": metadata.get("section"),
                        "position": metadata.get("position"),
                        "metadata": metadata,
                    })
            else:
                chunks.append({
                    "chunk_type": "text",
                    "content": content,
                    "image_path": None,
                    "page_number": metadata.get("page_number"),
                    "section": metadata.get("section"),
                    "position": metadata.get("position"),
                    "metadata": metadata,
                })

        elif content_type == "table":
            content = elem.get("content", "")
            if not content.strip():
                continue

            chunks.append({
                "chunk_type": "table",
                "content": content,
                "image_path": None,
                "page_number": metadata.get("page_number"),
                "section": metadata.get("section"),
                "position": metadata.get("position"),
                "metadata": metadata,
            })


        elif content_type == "image":
            chunks.append({
                "chunk_type": "image",
                "content": elem.get("content", "Extracted image from document."),
                "image_path": elem.get("image_path"),  
                "page_number": metadata.get("page_number"),
                "section": metadata.get("section"),
                "position": metadata.get("position"),
                "metadata": metadata,
            })

        else:
            # Ignore unknown types
            continue

    print(f"[ingestion] {len(chunks)} chunks ready for embedding")

  
    conn = get_db_conn()
    cur = conn.cursor()

    try:
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
            (pdf_path.name, str(pdf_path))
        )

        doc_id = str(cur.fetchone()["id"])
        print(f"[ingestion] Using document id={doc_id}")

        insert_sql = """
            INSERT INTO multimodal_chunks (
                doc_id,
                chunk_type,
                content,
                image_path,
                page_number,
                section,
                source_file,
                position,
                embedding,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        rows = []
        for c in chunks:
            embedding = _embeddings_model.embed_query(c["content"])

            rows.append((
                doc_id,
                c["chunk_type"],
                c["content"],
                c["image_path"],
                c["page_number"],
                c["section"],
                pdf_path.name,
                json.dumps(c["position"]) if c["position"] else None,
                embedding,
                json.dumps(c["metadata"]),
            ))

        execute_batch(cur, insert_sql, rows)
        conn.commit()

        print(f"[ingestion] Stored {len(rows)} chunks in multimodal_chunks")

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

    return {
        "status": "success",
        "document_id": doc_id,
        "chunks_ingested": len(rows),
    }



if __name__ == "__main__":
    pdfs = [
        "data/KB_Credit_Card_Spend_Summarizer.pdf"
    ]

    for pdf in pdfs:
        if pathlib.Path(pdf).exists():
            run_ingestion(pdf)
        else:
            print(f"PDF not found: {pdf}")