# ============================================================
# 统一异常定义 & 全局异常捕获
# ============================================================
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger


class AppException(HTTPException):
    """应用基类异常"""
    def __init__(self, status_code: int = 500, detail: str = "服务内部错误"):
        super().__init__(status_code=status_code, detail=detail)


class DocumentParseError(AppException):
    """文档解析异常"""
    def __init__(self, detail: str = "文档解析失败"):
        super().__init__(status_code=400, detail=detail)


class NotFoundError(AppException):
    """资源不存在"""
    def __init__(self, detail: str = "资源不存在"):
        super().__init__(status_code=404, detail=detail)


class UnauthorizedError(AppException):
    """权限不足"""
    def __init__(self, detail: str = "权限不足"):
        super().__init__(status_code=403, detail=detail)


class CircuitBreakerOpenError(AppException):
    """熔断器已打开"""
    def __init__(self, detail: str = "服务暂时不可用，请稍后重试"):
        super().__init__(status_code=503, detail=detail)


async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，统一返回格式"""
    if isinstance(exc, AppException):
        logger.warning(f"业务异常: {exc.detail} | path={request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail, "data": None}
        )
    # 未知异常
    logger.error(f"未捕获异常: {exc!r} | path={request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误", "data": None}
    )
