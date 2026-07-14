# ============================================================
# 企业智能知识库助手 - 主入口
# FastAPI 应用初始化、路由注册、中间件注册
# ============================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.config import get_settings
from backend.database import close_redis
from backend.exceptions import global_exception_handler
from backend.middleware import RequestLogMiddleware, SessionMiddleware
from backend.router import auth_router, upload_router, chat_router
from backend.router import knowledge_router, index_router, logs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    settings = get_settings()
    logger.info(f"{'='*50}")
    logger.info(f"  企业智能知识库助手 v1.0.0")
    logger.info(f"  服务启动: http://{settings.server_host}:{settings.server_port}")
    logger.info(f"{'='*50}")
    yield
    # 关闭资源
    await close_redis()
    logger.info("服务已关闭")


app = FastAPI(
    title="企业智能知识库助手",
    description="基于 RAG 的企业智能知识库问答系统，支持多格式文档解析、混合检索、流式输出",
    version="1.0.0",
    lifespan=lifespan,
)

# 异常处理
app.add_exception_handler(Exception, global_exception_handler)

# 中间件（顺序：从外到内）
app.add_middleware(SessionMiddleware)
app.add_middleware(RequestLogMiddleware)

# ---------- 注册路由 ----------
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(index_router)
app.include_router(logs_router)


# ---------- 前端静态文件 ----------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    """首页"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"message": "Frontend not built yet"})


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "企业智能知识库助手", "version": "1.0.0"}


@app.get("/api/eval/run")
async def run_evaluation(sample_count: int = 5, top_k: int = 5):
    """执行 RAG 评测"""
    from backend.tasks.evaluation import run_evaluation_task
    result = run_evaluation_task(top_k=top_k, sample_count=sample_count)
    return result


@app.get("/api/agents")
async def list_agents():
    """列出所有已注册 Agent"""
    from backend.services.rag_agent import agent_registry
    return {"code": 200, "data": agent_registry.list_agents()}


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
