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
            id,
            content,
            chunk_type,
            page_number,
            section,
            source_file
        FROM multimodal_chunks
        ORDER BY embedding <-> %(embedding)s::vector
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
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "search_type": "vector",
            }
        )
        for row in rows
    ]