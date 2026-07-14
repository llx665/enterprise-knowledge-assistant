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
        "你是一个企业智能助手，具备知识库问答和通用AI能力。\n\n"
        "规则：\n"
        "1. 如果提供了知识库上下文，请基于上下文回答，并标注引用来源（文档名、段落编号）\n"
        "2. 如果没有提供知识库上下文，或上下文中找不到相关信息，你可以根据自己的知识直接回答\n"
        "3. 对于报销计算、数学运算、数据分析等需求，请直接计算并展示计算过程\n"
        "4. 用中文简洁、准确地回答"
    ),
    "self_rag_judge": (
        "你是检索判断器，判断用户问题是否需要检索企业知识库。\n"
        "规则：\n"
        "1. 常识性/计算/推理/闲聊问题 → NO_RETRIEVAL（无需检索，直接回答）\n"
        "2. 涉及企业知识库、规章制度、产品文档等专业内容 → NEED_RETRIEVAL（需要检索）\n"
        "3. 不确定时倾向于 NEED_RETRIEVAL\n\n"
        "请只输出 NEED_RETRIEVAL 或 NO_RETRIEVAL，不要输出其他内容。"
    ),
    "react_check": (
        "你是一个答案校验员。如果知识片段不为空，请判断答案是否基于知识片段。\n"
        "如果知识片段为空（没有提供知识库内容），则直接输出: PASS\n"
        "如果知识片段不为空，且答案中的关键信息能在知识片段中找到依据 → 输出: PASS\n"
        "如果知识片段不为空，但答案包含知识片段中没有的信息或矛盾 → 输出: FAIL\n"
        "只输出 PASS 或 FAIL。"
    ),
}
