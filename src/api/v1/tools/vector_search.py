from typing import List
from langchain_core.documents import Document
from src.core.db import get_db_conn, get_embeddings


def query_documents(query: str, k: int = 25) -> List[Document]:
    """
    pgvector similarity search over multimodal_chunks.
    """
    embedder = get_embeddings()
    query_embedding = embedder.embed_query(query)

    sql = """
    SELECT
        c.id,
        c.content,
        c.chunk_type,
        c.image_path,
        c.page_number,
        c.section,
        c.source_file,
        d.created_at,
        d.updated_at
    FROM multimodal_chunks c
    JOIN documents d
    ON c.doc_id = d.id
    ORDER BY c.embedding <-> %(embedding)s::vector
    LIMIT %(k)s;

    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "embedding": query_embedding,
                    "k": k
                }
            )
            rows = cur.fetchall()

    return [
        Document(
            page_content=row["content"],
            metadata={
                "chunk_id": row["id"],
                "chunk_type": row["chunk_type"],
                "image_path": row["image_path"],          
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "created_date": row["created_at"],         
                "updated_date": row["updated_at"],        
                "search_type": "vector",
            }
        )
        for row in rows
    ]