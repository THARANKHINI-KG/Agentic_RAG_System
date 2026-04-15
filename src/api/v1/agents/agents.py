
import os
from typing import TypedDict, List, Literal

import cohere
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END

from src.api.v1.tools.fts_search import fts_search
from src.api.v1.tools.vector_search import query_documents
from src.api.v1.tools.hybrid_search import hybrid_search
from src.core.db import get_sql_database

load_dotenv(override=True)

# =============================================================================
# STATE
# =============================================================================
class RAGState(TypedDict):
    query: str
    original_query: str
    route: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    response: dict
    should_generate: bool
    retry_count: int
    generated_sql: str | None
    sql_result: str | None


# =============================================================================
# LLM helper
# =============================================================================
def _get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# =============================================================================
# ROUTER (DOCUMENT vs SQL)
# =============================================================================
class _RouteDecision(BaseModel):
    route: Literal["document", "sql"]
    reason: str
def router_node(state: RAGState) -> RAGState:
    print("\n[ROUTER NODE] Starting")

    llm = _get_llm().with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Decide how to answer the user's query.

Route to 'sql' if the query:
- Refers to the user's own data (e.g. "my", specific card IDs, customer IDs)
- Involves dates, months, billing periods
- Requires calculations, totals, comparisons, or breakdowns
- Needs exact numeric results from a database

Route to 'document' if the query:
- Asks for explanations, definitions, or scenarios
- Refers to documentation or example documents
- Is about how something works, not actual user data

Return a structured response with:
- route: either 'sql' or 'document'
- reason: a brief explanation of your decision
"""
        ),
        ("human", "Query: {query}")
    ])

    decision = (prompt | llm).invoke({"query": state["query"]})

    print(f"Router decision: {decision.route} ({decision.reason})")

    return {**state, "route": decision.route}


# =============================================================================
# SEARCH TOOLS
# =============================================================================
@tool
def fts_search_tool(query: str, k: int = 25):
    """Keyword-based full-text search over documents."""
    return fts_search(query, k)


@tool
def vector_search_tool(query: str, k: int = 25):
    """Vector similarity search over document chunks."""
    return query_documents(query, k)


@tool
def hybrid_search_tool(query: str, k: int = 25):
    """Hybrid keyword + vector search."""
    return hybrid_search(query, k)


TOOL_MAP = {
    "fts_search_tool": fts_search,
    "vector_search_tool": query_documents,
    "hybrid_search_tool": hybrid_search,
}

def generate_hyde_query(query: str) -> str:
    print("\n[HyDE] Generating hypothetical answer")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Write a concise, factual answer to the user's question. "
            "This text will be used ONLY to improve document retrieval."
        ),
        ("human", "{query}")
    ])

    response = (prompt | llm).invoke({"query": query})

    content = response.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )

    hyde_text = content.strip()

    print("[HyDE] Hypothetical document:")
    print(hyde_text)

    return hyde_text


def search_agent_node(state: RAGState) -> RAGState:
    print("\n[SEARCH AGENT NODE] Starting")

    llm = _get_llm().bind_tools([
        fts_search_tool,
        vector_search_tool,
        hybrid_search_tool,
    ])

    # ---------------------------------------------------------
    # Step 1: Normal tool-based retrieval
    # ---------------------------------------------------------
    resp = llm.invoke([
        ("system", "Choose the best retrieval tool. Return ONLY tool calls."),
        ("human", state["query"]),
    ])

    docs: List[Document] = []

    if getattr(resp, "tool_calls", None):
        call = resp.tool_calls[0]
        tool_name = call["name"]

        print(f"[SEARCH AGENT NODE] Tool selected: {tool_name}")

        fn = TOOL_MAP.get(tool_name)
        if fn:
            docs = fn(**call.get("args", {}))

    print(f"[SEARCH AGENT NODE] Docs from tool search: {len(docs)}")

    # ---------------------------------------------------------
    # Step 2: If docs found, return them
    # ---------------------------------------------------------
    if docs:
        print("[SEARCH AGENT NODE] ✅ Using tool-based retrieval results")
        return {
            **state,
            "retrieved_docs": docs,
            "should_generate": False,
        }

    # ---------------------------------------------------------
    # Step 3: HyDE fallback
    # ---------------------------------------------------------
    print("[SEARCH AGENT NODE] ⚠️ No docs found, triggering HyDE fallback")

    hyde_query = generate_hyde_query(state["query"])

    # Use HyDE text for hybrid search
    hyde_docs = hybrid_search(hyde_query, k=25)

    print(f"[SEARCH AGENT NODE] HyDE retrieval returned {len(hyde_docs)} docs")

    return {
        **state,
        "retrieved_docs": hyde_docs,
        "should_generate": False,
        "generated_sql": None,
    }
# =============================================================================
# RERANK
# =============================================================================
def rerank_node(state: RAGState) -> RAGState:
    print("\n[RERANK NODE] Starting")

    docs = state.get("retrieved_docs", [])
    if not docs:
        return {**state, "reranked_docs": []}

    primary_docs = []   # tables & images
    text_docs = []      # text only

    for d in docs:
        meta = d.metadata or {}
        if meta.get("chunk_type") in ("table", "image"):
            primary_docs.append(d)
        else:
            text_docs.append(d)

    reranked_text_docs = []

    if text_docs:
        co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        result = co.rerank(
            model="rerank-english-v3.0",
            query=state["query"],
            documents=[d.page_content for d in text_docs],
            top_n=min(5, len(text_docs)),
        )

        for r in result.results:
            base = text_docs[r.index]
            reranked_text_docs.append(
                Document(
                    page_content=base.page_content,
                    metadata={**base.metadata, "relevance_score": r.relevance_score},
                )
            )

    # ✅ PRIMARY FIRST: images + tables are never dropped
    reranked_docs = primary_docs + reranked_text_docs

    print(
        f"Reranked → primary (tables/images): {len(primary_docs)}, "
        f"text: {len(reranked_text_docs)}"
    )

    return {**state, "reranked_docs": reranked_docs}

# =============================================================================
# DECISION
# =============================================================================
def decision_node(state: RAGState) -> RAGState:
    print("\n[DECISION NODE] Starting")

    docs = state.get("reranked_docs", [])

    if not docs:
        print("No reranked docs → answer_absent")
        return {**state, "should_generate": False}

    # ✅ GENERIC TABLE OVERRIDE
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("chunk_type") == "table":
            print("Table detected → forcing generate")
            return {**state, "should_generate": True}

    # 🔁 FALLBACK TO LLM DECISION FOR NON-TABLE CONTENT
    llm = _get_llm()

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Decide whether the provided context contains information "
            "that can be used to answer the question.\n"
            "Respond ONLY with 'answer_present' or 'answer_absent'."
        ),
        (
            "human",
            "Question: {query}\n\nContext:\n{context}"
        )
    ])

    chain = prompt | llm
    raw = chain.invoke({
        "query": state["query"],
        "context": context,
    })

    verdict = raw.content.strip().lower()
    print(f"Decision verdict: {verdict}")

    return {**state, "should_generate": "answer_present" in verdict}

# =============================================================================
# REPHRASE (CRITICAL)
# =============================================================================
def rephrase_node(state: RAGState) -> RAGState:
    print("\n[REPHRASE NODE] Starting")

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the question to improve document retrieval. "
         "Preserve meaning."),
        ("human", "{query}")
    ])

    chain = prompt | llm
    raw = chain.invoke({"query": state["query"]})

    new_query = raw.content.strip()
    print(f"Rephrased query: {new_query}")

    return {
        **state,
        "query": new_query,
        "retry_count": state["retry_count"] + 1,
        "retrieved_docs": [],
        "reranked_docs": [],
    }


# =============================================================================
# SQL NODE
# =============================================================================

# =============================================================================
# HYBRID SQL NODE (SQL + RAG ENRICHMENT)
# =============================================================================
import re
from src.api.v1.tools.hybrid_search import hybrid_search

def _clean_sql(sql: str) -> str:
    import re

    if not sql:
        raise ValueError("Empty SQL")

    # Strip whitespace
    sql = sql.strip()

    # Remove markdown fences
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()

    # Remove leading SQL comments (-- or /* */)
    sql = re.sub(
        r"^\s*(--.*?\n|/\*.*?\*/\s*)+",
        "",
        sql,
        flags=re.DOTALL
    ).strip()

    # Extract the first keyword
    first_token = sql.split(None, 1)[0].lower()

    if first_token != "select":
        raise ValueError("Only SELECT queries are allowed")

    return sql.rstrip(";") + ";"


def _normalize_sql_rows(rows):
    """
    Convert SQLDatabase.run() output to JSON-safe values.
    """
    if not rows:
        return []

    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append({
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in row.items()
            })
        else:
            normalized.append(row)

    return normalized


def _requires_explanation(query: str) -> bool:
    """
    Decide if this query should be enriched with documents.
    """
    keywords = [
        "summarize", "summary", "explain", "why",
        "pattern", "trend", "insight", "what does this mean"
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def _summarize_sql_result(
    llm,
    user_query: str,
    sql: str,
    sql_rows: list,
    doc_context: str | None = None
) -> str:
    """
    Generic LLM-based explanation of SQL results,
    optionally enriched with document context.
    """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an analytical assistant.\n"
            "Use SQL results as the source of truth for numbers.\n"
            "Use documents ONLY to explain patterns or context.\n\n"
            "Rules:\n"
            "- Do NOT invent numbers\n"
            "- Do NOT contradict SQL results\n"
            "- Be concise (1–2 paragraphs max)\n"
            "- If data is empty, say so"
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "SQL query:\n{sql}\n\n"
            "SQL result:\n{result}\n\n"
            "Supporting documents (optional):\n{docs}"
        )
    ])

    raw = (prompt | llm).invoke({
        "question": user_query,
        "sql": sql,
        "result": sql_rows,
        "docs": doc_context or "No supporting documents provided."
    })

    content = raw.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )

    return content.strip()


def sql_node(state: RAGState) -> RAGState:
    print("\n[SQL NODE — HYBRID] Starting")

    llm = _get_llm()
    db = get_sql_database()

    # ---------------------------------------------------------
    # Step 1: Get schema
    # ---------------------------------------------------------
    schema = db.get_table_info()

    # ---------------------------------------------------------
    # Step 2: Generate SQL
    # ---------------------------------------------------------
    sql_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a PostgreSQL expert.\n"
        "Write a SAFE SELECT query using the schema below.\n\n"
        "Rules:\n"
        "- Return ONLY SQL (no markdown, no explanation)\n"
        "- Use only existing tables and columns from the schema\n"
        "- DO NOT generate INSERT, UPDATE, DELETE, DROP\n"
        "- Always return a SELECT query\n\n"
        "IMPORTANT SQL RULES:\n"
        "- If you use UNION or UNION ALL:\n"
        "  * Do NOT use expressions or functions in ORDER BY\n"
        "  * Either ORDER BY column names only\n"
        "  * OR wrap the UNION in a subquery and apply ORDER BY outside\n"
        "- Prefer subqueries when custom ordering or totals are required\n\n"
        "Schema:\n{schema}"
    ),
    ("human", "{query}")
    ])
    raw = (sql_prompt | llm).invoke({
        "schema": schema,
        "query": state["query"]
    })

    content = raw.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )

    sql = None
    try:
        sql = _clean_sql(content)
    except Exception as e:
        return {
            **state,
            "response": {
                "query": state["original_query"],
                "answer": f"SQL generation error: {e}",
                "retrieved_results": [],
                "sql_query": sql
            }
        }

    print(f"[SQL NODE] Executing SQL:\n{sql}")

    # ---------------------------------------------------------
    # Step 3: Execute SQL
    # ---------------------------------------------------------
    try:
        rows = db.run(sql)
    except Exception as e:
        return {
            **state,
            "response": {
                "query": state["original_query"],
                "answer": f"SQL execution error: {e}",
                "retrieved_results": [],
                "sql_query": sql
                
            }
        }

    normalized_rows = _normalize_sql_rows(rows)

    # ---------------------------------------------------------
    # Step 4: OPTIONAL RAG enrichment
    # ---------------------------------------------------------
    doc_context = None
    if _requires_explanation(state["original_query"]):
        print("[SQL NODE] 🔁 Enriching with document context")
        docs = hybrid_search(state["original_query"], k=5)
        doc_context = "\n\n".join(d.page_content for d in docs)

    # ---------------------------------------------------------
    # Step 5: Final hybrid explanation
    # ---------------------------------------------------------
    answer = _summarize_sql_result(
        llm=llm,
        user_query=state["original_query"],
        sql=sql,
        sql_rows=normalized_rows,
        doc_context=doc_context
    )

    # ---------------------------------------------------------
    # Step 6: Return CLEAN response (answer only)
    # ---------------------------------------------------------
    return {
        **state,
        "response": {
            "query": state["original_query"],
            "answer": answer,
            "retrieved_results": [],
            "sql_query": sql

        }
    }

# GENERATE ANSWER
# =============================================================================
def _serialize_retrieved_docs(docs: List[Document]) -> list:
    results = []

    for idx, doc in enumerate(docs):
        meta = doc.metadata or {}

        results.append({
            "chunk_id": meta.get("chunk_id", idx),
            "content": doc.page_content,
            "chunk_type": meta.get("chunk_type"),
            "page": meta.get("page_number"),
            "section": meta.get("section"),
            "source": meta.get("source_file"),
            "image_path": meta.get("image_path"),
            "similarity": meta.get("relevance_score"),
            "created_date": None,
            "updated_date": None,
        })

    return results
def generate_answer_node(state: RAGState) -> RAGState:
    print("\n[GENERATE ANSWER NODE — DOCUMENT FIRST HYBRID] Starting")

    # ---------------------------------------------------------
    # Safety: no documents available
    # ---------------------------------------------------------
    if not state["should_generate"] or not state.get("reranked_docs"):
        return {
            **state,
            "response": {
                "query": state["original_query"],
                "answer": (
                    "No relevant information was found in the available "
                    "documents to answer this query."
                ),
                "retrieved_results": [],
            },
        }

    docs = state["reranked_docs"]

    # ---------------------------------------------------------
    # Base document answer
    # ---------------------------------------------------------
    llm = _get_llm()
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are an assistant answering questions using documents.\n"
        "Base your response ONLY on the provided document content.\n"
        "Do not invent facts or numbers."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    raw = (prompt | llm).invoke({
        "context": context,
        "query": state["query"],
    })

    content = raw.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    document_answer = content.strip()

    # ---------------------------------------------------------
    # Detect need for SQL enrichment (document-first hybrid)
    # ---------------------------------------------------------
    def _requires_sql_enrichment(query: str) -> bool:
        keywords = [
            "how many", "total", "amount", "earned", "spent",
            "balance", "this month", "last month",
            "march", "april", "may", "june"
        ]
        q = query.lower()
        return any(k in q for k in keywords)

    sql_context = None

    if _requires_sql_enrichment(state["original_query"]):
        print("[GENERATE ANSWER NODE] ➕ SQL enrichment triggered")

        try:
            db = get_sql_database()
            schema = db.get_table_info()

            sql_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a PostgreSQL expert.\n"
                    "Write a SAFE SELECT query to fetch factual data that supports "
                    "the user's question.\n\n"
                    "Rules:\n"
                    "- SELECT only\n"
                    "- Use only existing schema\n"
                    "- Return ONLY SQL\n\n"
                    "Schema:\n{schema}"
                ),
                ("human", "{query}")
            ])

            raw_sql = (sql_prompt | llm).invoke({
                "schema": schema,
                "query": state["original_query"]
            })

            sql_text = raw_sql.content
            if isinstance(sql_text, list):
                sql_text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in sql_text
                )

            sql_text = _clean_sql(sql_text)

            rows = db.run(sql_text)
            sql_rows = _normalize_sql_rows(rows)

            sql_context = f"SQL facts:\n{sql_rows}"

        except Exception as e:
            print(f"[GENERATE ANSWER NODE] SQL enrichment skipped: {e}")

    # ---------------------------------------------------------
    # Merge document explanation + SQL facts
    # ---------------------------------------------------------
    if sql_context:
        merge_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant combining explanations with factual data.\n"
                "Rules:\n"
                "- Use document content for explanation and interpretation\n"
                "- Use SQL facts ONLY for numbers\n"
                "- Do NOT invent numbers\n"
                "- Keep the answer concise and clear"
            ),
            (
                "human",
                "Document Explanation:\n{doc}\n\n"
                "{sql}"
            )
        ])

        merged = (merge_prompt | llm).invoke({
            "doc": document_answer,
            "sql": sql_context
        })

        answer = merged.content.strip()
    else:
        answer = document_answer

    # ---------------------------------------------------------
    # Build retrieved_results metadata
    # ---------------------------------------------------------
    retrieved_results = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata or {}
        retrieved_results.append({
            "chunk_id": meta.get("chunk_id", idx),
            "content": doc.page_content,
            "chunk_type": meta.get("chunk_type"),
            "page": meta.get("page_number"),
            "section": meta.get("section"),
            "source": meta.get("source_file"),
            "image_path": meta.get("image_path"),
            "similarity": meta.get("relevance_score"),
            "created_date": None,
            "updated_date": None,
        })

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------
    return {
        **state,
        "response": {
            "query": state["original_query"],
            "answer": answer,
            "retrieved_results": retrieved_results,
        },
    }


# =============================================================================
# GRAPH
# =============================================================================
def route_after_decision(state: RAGState):
    if state["should_generate"]:
        return "generate"
    if state["retry_count"] < 2:
        return "rephrase"
    return "generate"


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router", router_node)
    graph.add_node("search", search_agent_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("decision", decision_node)
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("sql", sql_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"document": "search", "sql": "sql"},
    )

    graph.add_edge("search", "rerank")
    graph.add_edge("rerank", "decision")

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {"generate": "generate", "rephrase": "rephrase"},
    )

    graph.add_edge("rephrase", "search")
    graph.add_edge("generate", END)
    graph.add_edge("sql", END)

    return graph.compile()


# ✅ MUST BE AT MODULE LEVEL
rag_graph = build_rag_graph()


# =============================================================================
# ENTRYPOINT
# =============================================================================
def run_agent(user_query: str) -> dict:
    state: RAGState = {
        "query": user_query,
        "original_query": user_query,
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "should_generate": False,
        "retry_count": 0,
        "generated_sql": None,
        "sql_result": None,
    }

    final_state = rag_graph.invoke(state)
    return final_state["response"]