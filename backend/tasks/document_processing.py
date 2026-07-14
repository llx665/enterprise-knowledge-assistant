# ============================================================
# 文档异步处理任务
# 文档解析 → 文本清洗 → 向量入库，全部异步执行不阻塞问答接口
# ============================================================
import os
import time
from typing import List, Dict
from loguru import logger

from backend.services.parser import parser
from backend.services.vector_store import vector_store
from backend.services.bm25_retriever import bm25_retriever


def process_document_upload_task(
    file_path: str,
    filename: str,
    doc_id: str,
) -> Dict:
    """文档异步处理：解析 → 向量入库 → 更新 BM25 索引"""
    import asyncio

    start_time = time.time()
    result = {"doc_id": doc_id, "filename": filename, "status": "failed",
              "chunk_count": 0, "error": None, "duration_ms": 0}

    try:
        logger.info(f"开始异步处理文档: {filename} (id={doc_id})")
        parent_chunks = parser.parse(file_path, filename)
        if not parent_chunks:
            raise ValueError("文档解析结果为空")

        all_chunks = []
        for parent in parent_chunks:
            all_chunks.extend(parent["chunks"])

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            added = loop.run_until_complete(vector_store.add_documents(doc_id, all_chunks))
        finally:
            loop.close()

        bm25_chunks = [{"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata} for c in all_chunks]
        bm25_retriever.load_or_build(bm25_chunks)

        elapsed = (time.time() - start_time) * 1000
        result["status"] = "completed"
        result["chunk_count"] = added
        result["duration_ms"] = round(elapsed, 1)
        logger.info(f"文档处理完成: {filename} | {added} 块 | {elapsed:.1f}ms")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"文档处理失败: {filename}: {e}")
    return result


def rebuild_index_task() -> Dict:
    """重建所有索引"""
    import asyncio
    result = {"status": "failed", "error": None}
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(vector_store.rebuild_index())
        finally:
            loop.close()
        bm25_retriever.clear()
        result["status"] = "completed" if success else "failed"
        if not success:
            result["error"] = "向量索引重建失败"
    except Exception as e:
        result["error"] = str(e)
    return result


def delete_documents_task(doc_ids: List[str]) -> Dict:
    """删除文档异步任务"""
    import asyncio
    result = {"status": "failed", "deleted_count": 0, "error": None}
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            deleted = loop.run_until_complete(vector_store.delete_documents(doc_ids))
        finally:
            loop.close()
        result["status"] = "completed"
        result["deleted_count"] = deleted
    except Exception as e:
        result["error"] = str(e)
    return result
