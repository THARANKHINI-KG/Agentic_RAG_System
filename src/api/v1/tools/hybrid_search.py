from typing import List, Dict
from langchain_core.documents import Document
from src.api.v1.tools.vector_search import query_documents
from src.api.v1.tools.fts_search import fts_search


def hybrid_search(query: str, k: int = 25) -> List[Document]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).
    Deduplicates using multimodal_chunks.id (chunk_id).
    """
    vector_docs = query_documents(query, k=k)
    fts_docs = fts_search(query, k=k)

    RRF_K = 60
    scores: Dict[int, float] = {}
    docs: Dict[int, Document] = {}

    def key(doc: Document) -> int:
        return doc.metadata["chunk_id"]

    for rank, doc in enumerate(vector_docs):
        cid = key(doc)
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        docs[cid] = doc

    for rank, doc in enumerate(fts_docs):
        cid = key(doc)
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        docs.setdefault(cid, doc)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[cid] for cid, _ in ranked[:k]]