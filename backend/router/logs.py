# ============================================================
# 日志管理路由
# ============================================================
from fastapi import APIRouter, Query
from backend.models.schemas import ApiResponse, LogListResponse, LogQueryParams, LogStats
from backend.services.logger_service import logger_service

router = APIRouter(prefix="/api/logs", tags=["日志管理"])


@router.get("/list", response_model=LogListResponse)
async def get_logs(action: str = None, user: str = None, page: int = 1, page_size: int = 20):
    params = LogQueryParams(action=action, user=user, page=page, page_size=page_size)
    records, total = logger_service.query(params)
    return LogListResponse(data=records, total=total, page=page)


@router.get("/stats", response_model=ApiResponse)
async def get_stats():
    stats = logger_service.get_stats()
    return ApiResponse(data=stats)
