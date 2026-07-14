# ============================================================
# 混合检索链路模块
# Chroma 向量检索 + BM25 关键词检索 → RRF 加权融合 → BGE 重排
# 支持根据用户偏好调整检索权重
# ============================================================
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.config import get_settings
from backend.services.vector_store import vector_store
from backend.services.bm25_retriever import bm25_retriever
from backend.services.reranker import reranker
from backend.services.llm_service import llm_service


class HybridRetriever:
    """
    混合检索器
    向量检索 + BM25 → RRF 融合 → BGE 重排
    """

    def __init__(self):
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_keyword_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索主入口
        - 对查询进行向量检索和 BM25 检索
        - RRF 融合排序
        - BGE 重排（如果可用）
        - 返回 top_k 结果
        """
        # 1) 向量检索
        query_embedding = await llm_service.generate_embedding(query)
        vector_results = []
        if query_embedding:
            vector_results = await vector_store.similarity_search(
                query_embedding, top_k=self.settings.top_k_vector
            )

        # 2) BM25 关键词检索
        bm25_results = bm25_retriever.search(query, top_k=self.settings.top_k_bm25)

        # 3) 如果用户有关键词偏好，对 BM25 结果加权
        if user_keyword_weights:
            bm25_results = self._apply_user_weights(bm25_results, query, user_keyword_weights)

        # 4) RRF 融合排序
        fused = self._rrf_fusion(vector_results, bm25_results, k=self.settings.rrf_k)

        # 5) BGE 重排
        reranked = reranker.rerank(query, fused, top_k=self.settings.rerank_top_k)

        # 6) 截断到需要的数量
        final_results = reranked[:top_k]

        logger.info(
            f"混合检索完成: query={query[:30]}... | "
            f"vector={len(vector_results)}, bm25={len(bm25_results)}, "
            f"fused={len(fused)}, final={len(final_results)}"
        )
        return final_results

    def _rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 融合排序
        score = Σ 1 / (k + rank_i)
        """
        # 构建文档 ID → 分数的映射
        score_map = {}

        # 向量检索结果
        for rank, hit in enumerate(vector_results):
            chunk_id = hit.get("chunk_id", "")
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        # BM25 检索结果
        for rank, hit in enumerate(bm25_results):
            chunk_id = hit.get("chunk_id", "")
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        # 按 RRF 分数排序
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

        # 合并结果
        id_to_item = {}
        for hit in vector_results:
            id_to_item[hit.get("chunk_id", "")] = hit
        for hit in bm25_results:
            id_to_item[hit.get("chunk_id", "")] = hit

        fused = []
        for cid in sorted_ids:
            item = id_to_item.get(cid, {})
            item["rrf_score"] = score_map[cid]
            fused.append(item)

        return fused

    def _apply_user_weights(
        self,
        results: List[Dict],
        query: str,
        keyword_weights: Dict[str, float],
    ) -> List[Dict]:
        """根据用户历史偏好调整结果权重"""
        if not keyword_weights:
            return results

        for result in results:
            boost = 0.0
            content = (result.get("content", "") or "").lower()
            for kw, weight in keyword_weights.items():
                if kw.lower() in content:
                    boost += weight * 0.1  # 偏好加成
            if boost > 0:
                result["score"] = result.get("score", 0) * (1.0 + boost)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results


# 全局实例
hybrid_retriever = HybridRetriever()
