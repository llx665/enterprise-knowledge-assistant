# ============================================================
# Self-RAG 智能检索模块
# LLM 自主判断是否需要检索知识库，无相关知识直接回答
# ============================================================
from typing import List, Dict, Any, Optional, AsyncGenerator
from loguru import logger
import json

from backend.models.schemas import CitationSource
from backend.services.llm_service import llm_service, SYSTEM_PROMPTS
from backend.services.hybrid_retriever import hybrid_retriever
from backend.services.memory import short_memory, long_memory


_PLACEHOLDER_KEY = "sk-your-api-key-here"


def _check_api_key() -> bool:
    """检查 LLM API Key 是否已配置"""
    key = get_settings().llm_api_key
    return bool(key) and key != _PLACEHOLDER_KEY


def _api_key_error_msg() -> str:
    return (
        "请先配置 LLM API Key！\n\n"
        "编辑项目根目录的 .env 文件，设置有效的 LLM_API_KEY。\n"
        "支持 OpenAI、DeepSeek、Ollama 等兼容接口。\n"
        "示例：\n"
        "  LLM_API_KEY=sk-your-real-key\n"
        "  LLM_BASE_URL=https://api.openai.com/v1\n"
        "  或 Ollama: LLM_BASE_URL=http://localhost:11434/v1"
    )


class SelfRAG:
    """
    Self-RAG 智能检索器
    流程：判断是否需要检索 → 混合检索 → 生成回答 → 校验
    """

    async def judge_need_retrieval(self, query: str) -> bool:
        """
        判断是否需要检索知识库
        返回 True: 需要检索；False: 直接用 LLM 回答
        """
        result = await llm_service.chat(
            messages=[{"role": "user", "content": query}],
            system_prompt=SYSTEM_PROMPTS["self_rag_judge"],
        )
        need = "NEED_RETRIEVAL" in result.upper()
        logger.info(f"Self-RAG 判断: query={query[:30]}... → {'需要检索' if need else '无需检索'}")
        return need

    async def answer(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "visitor",
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """
        Self-RAG full flow.
        Generator yields SSE events: status, token, citation, done.
        """
        yield f"data: {json.dumps({'event': 'status', 'message': '正在分析问题...'})}\n\n"

        yield f"data: {json.dumps({'event': 'status', 'message': '正在检索知识库...'})}\n\n"

        # 获取用户长期偏好
        keyword_weights = long_memory.get_keyword_weights(user_id)
        history = short_memory.get_summary(session_id)

        # 混合检索
        results = await hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            user_keyword_weights=keyword_weights,
        )

        if not results:
            yield f"data: {json.dumps({'event': 'status', 'message': '知识库中未找到相关信息，正在直接回答...'})}\n\n"
            async for token in llm_service.chat_stream(
                messages=[{"role": "user", "content": query}],
                system_prompt=SYSTEM_PROMPTS["default"],
            ):
                yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"
            short_memory.add_message(session_id, "user", query)
            return

        # 3) 构建带上下文的 Prompt
        context_parts = []
        for r in results:
            content = r.get("content", "")[:500]
            meta = r.get("metadata", {})
            source = f"[来源: {meta.get('filename', '未知')}]"
            context_parts.append(f"{source}\n{content}")

        context = "\n\n---\n\n".join(context_parts)
        enhanced_prompt = (
            f"## 知识库上下文\n"
            f"{context}\n\n"
            f"## 用户问题\n{query}\n\n"
            f"请基于以上知识库上下文回答问题。如果上下文中没有相关信息，请明确说明。"
        )

        # 4) 生成回答（SSE 流式）
        yield f"data: {json.dumps({'event': 'status', 'message': '正在生成回答...'})}\n\n"

        full_answer = ""
        async for token in llm_service.chat_stream(
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPTS['default']}\n会话摘要: {history}" if history else SYSTEM_PROMPTS["default"]},
                {"role": "user", "content": enhanced_prompt},
            ],
        ):
            full_answer += token
            yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"

        # 5) 构建引用来源
        citations = []
        for r in results:
            meta = r.get("metadata", {})
            page = r.get("page_number") or meta.get("page_number", "")
            para = r.get("paragraph_number") or meta.get("paragraph_number", "")

            citation = CitationSource(
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                content=r.get("content", "")[:200],
                page_number=int(page) if page and str(page).isdigit() else None,
                paragraph_number=int(para) if para and str(para).isdigit() else None,
                score=float(r.get("rrf_score", r.get("score", 0))),
            )
            citations.append(citation)

        # 发送引用
        yield f"data: {json.dumps({'event': 'citation', 'citations': [c.model_dump() for c in citations]})}\n\n"

        # 6) 记录短期和长期记忆
        short_memory.add_message(session_id, "user", query)
        short_memory.add_message(session_id, "assistant", full_answer[:200])

        # 提取关键词记录长期偏好
        import jieba
        keywords = list(jieba.cut(query))[:10]
        long_memory.record_search(user_id, query, keywords)

        # 发送完成事件（含校验信息）
        yield f"data: {json.dumps({'event': 'done', 'need_retrieval': True})}\n\n"

    async def validate_answer(
        self, query: str, answer: str, context: str
    ) -> bool:
        """
        ReAct 自省：验证回答是否基于知识库内容
        """
        check_prompt = (
            f"## 知识片段\n{context[:1000]}\n\n"
            f"## 答案\n{answer[:500]}\n\n"
            f"请判断答案是否基于知识片段生成。"
        )
        result = await llm_service.chat(
            messages=[{"role": "user", "content": check_prompt}],
            system_prompt=SYSTEM_PROMPTS["react_check"],
        )
        passed = "PASS" in result.upper()
        logger.info(f"ReAct 校验: query={query[:20]}... → {'通过' if passed else '失败'}")
        return passed


# 全局实例
self_rag = SelfRAG()
