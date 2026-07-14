# ============================================================
# BM25 关键词检索模块
# 使用 rank-bm25 + jieba 分词实现中文关键词匹配检索
# ============================================================
import os
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.config import get_settings


class BM25Retriever:
    """
    BM25 关键词检索器
    构建文档的倒排索引，支持中文分词
    """

    def __init__(self):
        self.settings = get_settings()
        self._corpus: List[Dict] = []       # 文档列表: [{chunk_id, content, metadata}]
        self._bm25 = None
        self._tokenizer = None
        self._is_built = False
        self.index_path = os.path.join(
            os.path.dirname(self.settings.chroma_persist_dir), "bm25_index.json"
        )

    def _get_tokenizer(self):
        """获取分词器（延迟加载 jieba）"""
        if self._tokenizer is None:
            import jieba
            # 精简 jieba 日志
            jieba.setLogLevel(20)
            self._tokenizer = jieba
        return self._tokenizer

    def _tokenize(self, text: str) -> List[str]:
        """对文本做中文分词"""
        tokenizer = self._get_tokenizer()
        return list(tokenizer.cut(text))

    def build_index(self, chunks: List[Dict]):
        """
        构建 BM25 索引
        chunks: [{"chunk_id": str, "content": str, "metadata": dict}, ...]
        """
        from rank_bm25 import BM25Okapi

        self._corpus = chunks
        if not chunks:
            self._is_built = False
            self._bm25 = None
            return

        # 分词
        tokenized_corpus = [self._tokenize(c["content"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._is_built = True

        # 持久化（简化存储：仅存文本，运行时重建）
        self._persist_index(chunks)
        logger.info(f"BM25 索引构建完成: {len(chunks)} 文档")

    def load_or_build(self, all_chunks: List[Dict]) -> bool:
        """
        加载或重建 BM25 索引
        返回是否成功加载
        """
        if all_chunks:
            self.build_index(all_chunks)
            return True
        return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 关键词检索"""
        if not self._is_built or self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 排序取 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                continue
            chunk = self._corpus[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": float(score),
                "retriever": "bm25",
            })
        return results

    def _persist_index(self, chunks: List[Dict]):
        """持久化索引数据（仅存文本，tokenizer 运行时重建）"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            # 只存必要字段，减少存储
            simplified = [
                {"chunk_id": c["chunk_id"], "content": c["content"][:500],
                 "metadata": c.get("metadata", {})}
                for c in chunks
            ]
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(simplified, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"BM25 索引持久化失败: {e}")

    def clear(self):
        """清除索引"""
        self._corpus = []
        self._bm25 = None
        self._is_built = False
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception:
                pass


# 全局实例
bm25_retriever = BM25Retriever()
