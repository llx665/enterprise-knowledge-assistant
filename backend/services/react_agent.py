# ============================================================
# ReAct 自省模块
# 回答完成后自动校验是否匹配知识库，不匹配则重新检索修正
# ============================================================
from typing import Tuple, Optional
from loguru import logger

from backend.services.self_rag import self_rag
from backend.services.hybrid_retriever import hybrid_retriever
from backend.services.llm_service import llm_service, SYSTEM_PROMPTS
from backend.services.memory import short_memory


class ReActAgent:
    """
    ReAct 自省 Agent
    回答完成后自动执行:
    1. 校验回答是否基于知识库内容
    2. 如果校验失败，重新检索并修正回答
    3. 最多重试 2 轮
    """

    MAX_RETRIES = 2

    async def process(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
    ) -> Tuple[str, bool]:
        """
        执行 ReAct 流程
        返回 (最终答案, 是否通过校验)
        """
        history = short_memory.get_summary(session_id)
        context_parts = []

        for attempt in range(self.MAX_RETRIES + 1):
            # 检索
            results = await hybrid_retriever.retrieve(query=query, top_k=top_k)

            context = ""
            if results:
                context_parts = [f"[{r.get('metadata', {}).get('filename', '未知')}]\n{r.get('content', '')[:300]}"
                                 for r in results]
                context = "\n\n".join(context_parts)

            # 生成回答
            if attempt > 0:
                instruction = ("请重新审视问题，参考以下知识库内容给出准确回答。"
                               "注意：上次回答可能不正确，请务必基于知识库内容。")
            else:
                instruction = "请基于知识库内容回答。"

            prompt = (
                f"## 知识库上下文\n{context}\n\n"
                f"## 用户问题\n{query}\n\n"
                f"{instruction}"
            ) if context else query

            system = (f"{SYSTEM_PROMPTS['default']}\n会话摘要: {history}" if history
                      else SYSTEM_PROMPTS["default"])

            answer = await llm_service.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system,
            )

            if not answer:
                return "抱歉，暂时无法回答该问题。", False

            # 校验
            passed = await self_rag.validate_answer(query, answer, context)
            logger.info(f"ReAct 校验 第{attempt + 1}轮: {'通过' if passed else '未通过'}")

            if passed:
                return answer, True

        # 超过重试次数，返回最后一次回答
        logger.warning(f"ReAct 重试耗尽，返回最后回答: query={query[:20]}")
        return answer, False

    async def check_and_correct(
        self, query: str, answer: str, context: str
    ) -> Tuple[str, bool]:
        """单次校验纠错"""
        passed = await self_rag.validate_answer(query, answer, context)
        if passed:
            return answer, True

        # 重新检索并修正
        logger.info("ReAct 修正: 重新检索并生成回答")
        return await self.process(query)


# 全局实例
react_agent = ReActAgent()
