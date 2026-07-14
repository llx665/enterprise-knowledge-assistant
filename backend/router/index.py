# ============================================================
# 索引管理路由
# ============================================================
from fastapi import APIRouter
from loguru import logger
from backend.models.schemas import ApiResponse, IndexInfoResponse
from backend.services.vector_store import vector_store
from backend.services.logger_service import logger_service
from backend.config import get_settings

router = APIRouter(prefix="/api/index", tags=["索引管理"])


@router.get("/info", response_model=IndexInfoResponse)
async def get_index_info():
    info = await vector_store.get_index_info()
    return IndexInfoResponse(data=info)


@router.post("/rebuild", response_model=ApiResponse)
async def rebuild_index():
    try:
        import redis as sync_redis
        from rq import Queue
        settings = get_settings()
        sync_conn = sync_redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, password=settings.redis_password or None)
        q = Queue("default", connection=sync_conn)
        from backend.tasks.document_processing import rebuild_index_task
        q.enqueue(rebuild_index_task)
    except Exception:
        from backend.tasks.document_processing import rebuild_index_task
        rebuild_index_task()
    logger_service.log(action="rebuild", user="system", detail="重建索引")
    return ApiResponse(message="已提交重建任务")
