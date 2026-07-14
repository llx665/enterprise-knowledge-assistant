# ============================================================
# LLM 服务模块
# 封装 OpenAI 兼容接口调用，支持 SSE 流式输出
# 集成熔断与重试机制
# ============================================================
from typing import AsyncGenerator, List, Dict, Optional
from openai import AsyncOpenAI
from loguru import logger
import json

from backend.config import get_settings
from backend.services.circuit_breaker import llm_cb, embed_cb, with_retry


class LLMService:
    """大语言模型服务"""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    @with_retry(max_attempts=3)
    async def chat_stream(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        SSE 流式对话
        逐 token 产出文本，支持中断
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        async def _call():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

        try:
            stream = await llm_cb.call(_call)
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"LLM 流式请求失败: {e}")
            yield f"\n\n[LLM 调用异常: {e}]"

    @with_retry(max_attempts=3)
    async def chat(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> str:
        """非流式对话（用于内部调用，如 Self-RAG 判断）"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        async def _call():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )

        try:
            response = await llm_cb.call(_call)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM 对话请求失败: {e}")
            return ""

    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """简单的字符级 hash 嵌入（当 sentence-transformers 不可用时）"""
        import numpy as np
        vec = np.zeros(dim)
        for i, ch in enumerate(text):
            idx = (ord(ch) * 2654435761 + i * 314159) % dim
            vec[idx] += 0.1
            vec[(idx + 1) % dim] += 0.05
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def generate_embedding(self, text: str) -> List[float]:
        """生成文本向量，优先使用本地模型，降级使用简单嵌入"""
        try:
            # 优先使用本地模型（sentence-transformers）
            from sentence_transformers import SentenceTransformer
            settings = get_settings()
            model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
            vec = model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"本地模型不可用，使用简单嵌入降级: {e}")
            return self._simple_embedding(text)


# 全局 LLM 服务实例
llm_service = LLMService()


# ---------- 系统 Prompt 模板 ----------

SYSTEM_PROMPTS = {
    "default": (
        "你是一个专业的企业知识库助手。请你基于提供的知识片段，"
        "用中文简洁、准确地回答用户问题。如果知识片段中找不到答案，"
        "请直接说明「知识库中没有相关信息」，不要编造答案。"
        "引用来源时请标注原文片段和文档名。"
    ),
    "self_rag_judge": (
        "你是一个知识库检索判断器。你需要判断用户的问题是否需要检索企业知识库来回答。\n"
        "规则：\n"
        "1. 如果问题是常识性问题（如「你好」、「今天天气」、「1+1等于几」）→ 输出: NO_RETRIEVAL\n"
        "2. 如果问题涉及企业知识库、技术文档、规章制度等专业内容 → 输出: NEED_RETRIEVAL\n"
        "3. 如果用户只是打招呼或闲聊 → 输出: NO_RETRIEVAL\n\n"
        "请只输出 NEED_RETRIEVAL 或 NO_RETRIEVAL，不要输出其他内容。"
    ),
    "react_check": (
        "你是一个答案校验员。请判断以下答案是否基于提供的知识片段。\n"
        "如果答案中的关键信息都能在知识片段中找到依据 → 输出: PASS\n"
        "如果答案包含知识片段中没有的信息或矛盾 → 输出: FAIL\n"
        "只输出 PASS 或 FAIL。"
    ),
}
