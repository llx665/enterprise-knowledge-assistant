# ============================================================
# 数据库 & 缓存连接管理
# 管理 Redis 连接池、ChromaDB 客户端等全局资源
# ============================================================
import redis.asyncio as aioredis
import chromadb
from chromadb.config import Settings as ChromaSettings
from functools import lru_cache
from loguru import logger

from backend.config import get_settings

_redis_pool = None
_chroma_client = None


async def get_redis() -> aioredis.Redis:
    """获取 Redis 连接（带连接池）"""
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            max_connections=20,
        )
        logger.info(f"Redis 连接池已创建: {settings.redis_host}:{settings.redis_port}")
    return aioredis.Redis(connection_pool=_redis_pool)


@lru_cache()
def get_chroma_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 持久化客户端（单例）"""
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB 客户端已创建: {settings.chroma_persist_dir}")
    return _chroma_client


async def close_redis():
    """关闭 Redis 连接池"""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("Redis 连接池已关闭")
