# ============================================================
# 向量索引模块
# 使用 ChromaDB 做文档向量化入库、增量更新、版本回滚、索引清理
# ============================================================
import uuid
import time
import json
import os
from typing import List, Dict, Any, Optional
from loguru import logger
import chromadb.utils.embedding_functions as embedding_functions

from backend.config import get_settings
from backend.database import get_chroma_client
from backend.models.schemas import ChunkInfo, IndexVersion
from backend.services.circuit_breaker import vector_cb


class VectorStoreService:
    """ChromaLDB 向量存储服务"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_chroma_client()
        self.collection_name = self.settings.chroma_collection_name
        self._collection = None
        # 索引版本管理文件
        self.version_file = os.path.join(
            self.settings.chroma_persist_dir, "index_versions.json"
        )

    def _get_or_create_collection(self):
        """获取或创建向量集合"""
        if self._collection is not None:
            return self._collection
        try:
            # 尝试获取已有集合
            self._collection = self.client.get_collection(self.collection_name)
        except Exception:
            # 创建新集合
            self._collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "企业知识库向量索引"},
            )
            logger.info(f"创建向量集合: {self.collection_name}")
        return self._collection

    async def add_documents(self, document_id: str, chunks: List[ChunkInfo]) -> int:
        """
        将文档分块向量化并入库
        返回入库的块数
        """
        from backend.services.llm_service import llm_service as llm_svc

        collection = self._get_or_create_collection()
        batch_size = 50  # ChromaDB 批量写入大小

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = []
            texts = []
            embeddings = []
            metadatas = []

            for chunk in batch:
                ids.append(chunk.chunk_id)
                texts.append(chunk.content)
                # 生成向量
                from backend.services.llm_service import llm_service as llm_svc
                emb = await llm_svc.generate_embedding(chunk.content)
                embeddings.append(emb if emb else [0.0] * 384)
                metadatas.append({
                    "doc_id": document_id,
                    "parent_id": chunk.parent_id,
                    "filename": chunk.metadata.get("filename", ""),
                    "page_number": str(chunk.page_number or ""),
                    "paragraph_number": str(chunk.paragraph_number or ""),
                    "file_type": chunk.metadata.get("file_type", ""),
                    "chunk_index": str(chunk.metadata.get("child_idx", "")),
                })

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                added += len(batch)
            except Exception as e:
                logger.error(f"向量入库失败 (批次 {i}): {e}")
                raise

        # 记录索引版本
        self._record_version(document_id, len(chunks))

        logger.info(f"向量入库完成: document={document_id}, chunks={added}")
        return added

    async def similarity_search(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """向量相似度检索"""
        collection = self._get_or_create_collection()

        async def _search():
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            return results

        try:
            results = await vector_cb.call(_search)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "chunk_id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0),
                    "retriever": "vector",
                })
        return hits

    async def delete_documents(self, doc_ids: List[str]) -> int:
        """删除指定文档的向量"""
        collection = self._get_or_create_collection()
        deleted = 0
        for doc_id in doc_ids:
            try:
                collection.delete(where={"doc_id": doc_id})
                deleted += 1
                logger.info(f"删除文档向量: {doc_id}")
            except Exception as e:
                logger.error(f"删除文档向量失败 {doc_id}: {e}")
        return deleted

    async def rebuild_index(self):
        """重建索引（删除旧集合，创建新集合）"""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            self._get_or_create_collection()
            # 清除版本记录
            if os.path.exists(self.version_file):
                os.remove(self.version_file)
            logger.info("向量索引已重建")
            return True
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False

    async def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        collection = self._get_or_create_collection()
        try:
            count = collection.count()
        except Exception:
            count = 0

        versions = self._load_versions()
        return {
            "collection_name": self.collection_name,
            "chunk_count": count,
            "version_count": len(versions),
            "versions": versions,
        }

    # ---------- 索引版本管理 ----------

    def _record_version(self, doc_id: str, chunk_count: int):
        """记录索引版本快照"""
        versions = self._load_versions()
        versions.append({
            "version_id": str(uuid.uuid4())[:8],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "doc_id": doc_id,
            "chunk_count": chunk_count,
        })
        # 最多保留 50 个版本记录
        if len(versions) > 50:
            versions = versions[-50:]
        self._save_versions(versions)

    def _load_versions(self) -> List[Dict]:
        """加载索引版本记录"""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"索引版本文件加载失败: {e}")
        return []

    def _save_versions(self, versions: List[Dict]):
        """保存索引版本记录"""
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        try:
            with open(self.version_file, "w", encoding="utf-8") as f:
                json.dump(versions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"索引版本文件保存失败: {e}")


# 全局实例
vector_store = VectorStoreService()
