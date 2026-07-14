# ============================================================
# FastAPI 中间件
# 包括请求日志、性能统计、权限校验等
# ============================================================
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestLogMiddleware(BaseHTTPMiddleware):
    """请求日志中间件：记录每个请求的方法、路径、耗时"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()
        method = request.method
        path = request.url.path

        logger.info(f"[{request_id}] → {method} {path}")

        try:
            response: Response = await call_next(request)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"[{request_id}] ✗ {method} {path} | {type(e).__name__} | {elapsed:.1f}ms")
            raise

        elapsed = (time.time() - start) * 1000
        # 慢请求警告（超过 3 秒）
        if elapsed > 3000:
            logger.warning(f"[{request_id}] 慢请求: {method} {path} | {elapsed:.1f}ms")
        else:
            logger.info(f"[{request_id}] ← {method} {path} | {response.status_code} | {elapsed:.1f}ms")

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round(elapsed, 1))
        return response


class SessionMiddleware(BaseHTTPMiddleware):
    """会话管理中间件：确保每个请求携带 session_id"""

    async def dispatch(self, request: Request, call_next):
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:12]}"
        request.state.session_id = session_id
        request.state.user_agent = request.headers.get("User-Agent", "unknown")
        return await call_next(request)
