# ============================================================
# ??????
# ============================================================
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import get_settings
from backend.services.bm25_retriever import bm25_retriever
from backend.services.reranker import reranker


def test_bm25_build_and_search():
    chunks = [
        {"chunk_id": "c1", "content": "Python is a programming language", "metadata": {"filename": "doc1"}},
        {"chunk_id": "c2", "content": "Company attendance policy", "metadata": {"filename": "doc2"}},
        {"chunk_id": "c3", "content": "Python decorators modify functions", "metadata": {"filename": "doc1"}},
    ]
    bm25_retriever.build_index(chunks)
    results = bm25_retriever.search("Python function", top_k=5)
    assert len(results) > 0
    assert any("Python" in r["content"] for r in results)
    print(f"PASS: BM25 search -> {len(results)} results")


def test_bm25_no_match():
    chunks = [{"chunk_id": "c1", "content": "Python programming", "metadata": {}}]
    bm25_retriever.build_index(chunks)
    results = bm25_retriever.search("nonexistent", top_k=5)
    assert len(results) == 0 or all(r["score"] <= 0 for r in results)
    print("PASS: BM25 no-match")


def test_rrf_fusion():
    from backend.services.hybrid_retriever import hybrid_retriever
    v = [{"chunk_id": "c1", "content": "A", "metadata": {}, "score": 0.9},
         {"chunk_id": "c2", "content": "B", "metadata": {}, "score": 0.8}]
    b = [{"chunk_id": "c2", "content": "B", "metadata": {}, "score": 0.7},
         {"chunk_id": "c3", "content": "C", "metadata": {}, "score": 0.6}]
    fused = hybrid_retriever._rrf_fusion(v, b, k=60)
    assert len(fused) >= 2
    print(f"PASS: RRF fusion -> {len(fused)} results")


if __name__ == "__main__":
    print("Running retrieval tests...")
    test_bm25_build_and_search()
    test_bm25_no_match()
    test_rrf_fusion()
    print("All retrieval tests passed!")
