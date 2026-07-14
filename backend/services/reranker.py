# ============================================================
# 重排序模块
# 使用 BGE Reranker 模型对检索结果做精排
# ============================================================
from typing import List, Dict, Any
from loguru import logger

from backend.config import get_settings


class RerankerService:
    """
    重排序服务
    使用 BAAI/bge-reranker-v2-m3 对召回结果做交叉编码排序
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """延迟加载重排序模型"""
        if self._model_loaded:
            return self._model
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = self.settings.rerank_model
            logger.info(f"加载重排序模型: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            self._model_loaded = True
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.warning(f"重排序模型加载失败（将使用 RRF 排序兜底）: {e}")
            self._model_loaded = False
        return self._model

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对检索结果重排序
        results: [{"content": str, "score": float, ...}, ...]
        返回重排后的前 top_k 结果
        """
        if not results:
            return []

        model = self._load_model()
        if model is None:
            # 降级：按原分数排序
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:top_k]

        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            pairs = [[query, r["content"][:512]] for r in results]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(device)

            with torch.no_grad():
                scores = model(**inputs).logits.view(-1).float().cpu().numpy()

            # 合并分数并排序
            for i, r in enumerate(results):
                r["rerank_score"] = float(scores[i])

            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        except Exception as e:
            logger.warning(f"重排序计算失败（降级使用原分数排序）: {e}")
            results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return results[:top_k]


# 全局实例
reranker = RerankerService()
