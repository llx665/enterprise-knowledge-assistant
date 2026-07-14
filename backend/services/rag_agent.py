# ============================================================
# 多 Agent 扩展预留模块
# 提供检索 Agent、溯源校验 Agent 接口，方便后续升级多智能体架构
# ============================================================
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from loguru import logger

from backend.models.schemas import CitationSource, SearchQuery, SearchResponse
from backend.services.self_rag import self_rag


# ---------- Agent 抽象基类 ----------

class BaseAgent(ABC):
    """Agent 抽象基类，所有 Agent 必须实现 run 方法"""

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        ...

    @property
    @abstractmethod
    def agent_name(self) -> str:
        ...


# ---------- 检索 Agent ----------

class RetrievalAgent(BaseAgent):
    """
    检索 Agent
    负责：判断是否检索 → 混合检索 → 返回上下文
    """

    @property
    def agent_name(self) -> str:
        return "检索Agent"

    async def run(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "visitor",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        执行检索任务
        返回: {
            "need_retrieval": bool,
            "results": List[Dict],
            "context": str,
            "citations": List[CitationSource],
        }
        """
        from backend.services.hybrid_retriever import hybrid_retriever
        from backend.services.memory import long_memory

        # 1) 判断是否需要检索
        need_retrieval = await self_rag.judge_need_retrieval(query)

        if not need_retrieval:
            return {
                "need_retrieval": False,
                "results": [],
                "context": "",
                "citations": [],
            }

        # 2) 混合检索（含用户偏好）
        keyword_weights = long_memory.get_keyword_weights(user_id)
        results = await hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            user_keyword_weights=keyword_weights,
        )

        # 3) 构建上下文 & 引用
        context_parts = []
        citations = []
        for r in results:
            meta = r.get("metadata", {})
            content = r.get("content", "")[:500]
            context_parts.append(f"[{meta.get('filename', '未知')}]\n{content}")
            citations.append(CitationSource(
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                content=content[:200],
                page_number=meta.get("page_number"),
                paragraph_number=meta.get("paragraph_number"),
                score=float(r.get("rrf_score", r.get("score", 0))),
            ))

        return {
            "need_retrieval": True,
            "results": results,
            "context": "\n\n---\n\n".join(context_parts),
            "citations": citations,
        }


# ---------- 溯源校验 Agent ----------

class SourceVerifyAgent(BaseAgent):
    """
    溯源校验 Agent
    负责：验证回答是否基于真实知识库内容，检测幻觉
    """

    @property
    def agent_name(self) -> str:
        return "溯源校验Agent"

    async def run(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        执行溯源校验
        返回: {
            "is_grounded": bool,
            "check_details": List[str],
            "confidence": float,
        }
        """
        from backend.services.llm_service import llm_service, SYSTEM_PROMPTS

        # 使用 LLM 校验
        check_prompt = (
            f"知识库片段:\n{context[:2000]}\n\n"
            f"生成的答案:\n{answer[:500]}\n\n"
            f"答案中的每个关键信息点在知识库中是否有依据？请分析并给出结论。"
        )

        result = await llm_service.chat(
            messages=[{"role": "user", "content": check_prompt}],
            system_prompt=SYSTEM_PROMPTS["react_check"],
        )

        passed = "PASS" in result.upper()

        return {
            "is_grounded": passed,
            "check_details": [result],
            "confidence": 0.9 if passed else 0.3,
        }


# ---------- Agent 注册与管理 ----------

class AgentRegistry:
    """Agent 注册中心，方便后续扩展多智能体架构"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.agent_name] = agent
        logger.info(f"Agent 已注册: {agent.agent_name}")

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())


# 全局注册中心 & 默认 Agent 实例
agent_registry = AgentRegistry()
retrieval_agent = RetrievalAgent()
source_verify_agent = SourceVerifyAgent()

# 注册默认 Agent
agent_registry.register(retrieval_agent)
agent_registry.register(source_verify_agent)
