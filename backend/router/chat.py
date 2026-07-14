# ============================================================
# 问答聊天路由
# SSE 流式输出、ReAct 自省、记忆联动
# ============================================================
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.models.schemas import SearchQuery
from backend.services.self_rag import self_rag
from backend.services.react_agent import react_agent
from backend.services.logger_service import logger_service
from backend.services.memory import long_memory

router = APIRouter(prefix="/api/chat", tags=["问答聊天"])


@router.post("/ask")
async def ask_question(query: SearchQuery, request: Request):
    """
    SSE 流式问答接口
    支持 Self-RAG 智能检索 + 流式输出 + 引用来源
    """
    session_id = query.session_id or request.state.session_id
    user = getattr(request.state, "user", None) or request.headers.get("X-User-ID", "visitor")
    user_id = user if isinstance(user, str) else getattr(user, "username", "visitor")

    start_time = time.time()

    async def event_generator():
        full_answer = ""
        context_snippets = []

        try:
            async for event in self_rag.answer(
                query=query.query,
                session_id=session_id,
                user_id=user_id,
                top_k=query.top_k,
            ):
                yield event

                # 收集流式 token
                try:
                    data = json.loads(event[6:])  # 去掉 "data: "
                    if data.get("event") == "token":
                        full_answer += data.get("content", "")
                    elif data.get("event") == "citation":
                        context_snippets = data.get("citations", [])
                except (json.JSONDecodeError, IndexError):
                    pass

            # ReAct 自省校验（非流式）
            if full_answer and context_snippets:
                context_text = "\n".join([c.get("content", "") for c in context_snippets[:3]])
                passed = await react_agent.check_and_correct(
                    query.query, full_answer, context_text
                )

            # 记录日志
            elapsed = (time.time() - start_time) * 1000
            logger_service.log(
                action="search",
                user=user_id,
                detail=f"会话: {session_id[:8]}",
                duration_ms=elapsed,
                query=query.query,
                recall_count=len(context_snippets),
            )

        except Exception as e:
            logger.error(f"问答流异常: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ask/sync")
async def ask_sync(query: SearchQuery, request: Request):
    """
    同步问答接口（非流式）
    返回完整回答 + 引用 + 校验结果
    """
    session_id = query.session_id or request.state.session_id
    user_id = getattr(request.state, "user", "visitor")

    start_time = time.time()

    try:
        answer, passed = await react_agent.process(
            query=query.query, session_id=session_id, top_k=query.top_k
        )

        elapsed = (time.time() - start_time) * 1000
        logger_service.log(
            action="search", user=user_id,
            detail=f"同步问答 | 校验: {'通过' if passed else '失败'}",
            duration_ms=elapsed, query=query.query,
        )

        return {
            "code": 200,
            "message": "success",
            "data": {
                "answer": answer,
                "self_check_passed": passed,
            },
        }
    except Exception as e:
        logger.error(f"同步问答异常: {e}")
        return {"code": 500, "message": str(e), "data": None}


@router.post("/history")
async def get_history(request: Request, session_id: str = "default"):
    """获取会话历史"""
    from backend.services.memory import short_memory
    history = short_memory.get_history(session_id)
    return {"code": 200, "data": history}


@router.post("/clear")
async def clear_history(session_id: str = "default"):
    """清除会话历史"""
    from backend.services.memory import short_memory
    short_memory.clear_session(session_id)
    return {"code": 200, "message": "会话已清除"}


@router.get("/hot-queries")
async def get_hot_queries(top_k: int = 10):
    """获取热门查询"""
    stats = logger_service.get_stats()
    return {"code": 200, "data": stats.get("top_queries", [])[:top_k]}
