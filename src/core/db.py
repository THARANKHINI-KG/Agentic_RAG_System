import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase

load_dotenv(override=True)


def get_db_conn():
    """
    Returns a psycopg2 connection with dict-style rows.
    This is used for:
    - ingestion inserts
    - pgvector similarity search
    - full-text search
    """
    db_url = os.getenv("AGENTIC_RAG_PG_DSN")
    if not db_url:
        raise ValueError("AGENTIC_RAG_PG_DSN is not set")

    return psycopg2.connect(
        db_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_embeddings():
    """
    Returns a Gemini embedding model.
    """
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDINGS_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        output_dimensionality=1536,
    )



def get_sql_database():
    """
    Returns a LangChain SQLDatabase instance.
    Used by the SQL agent for:
    - schema inspection (get_table_info)
    - safe SELECT execution (run)
    """
    db_url = os.getenv("AGENTIC_RAG_DB_URL")
    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set")

    return SQLDatabase.from_uri(
        db_url,
        sample_rows_in_table_info=2,  
    )