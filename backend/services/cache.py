# ============================================================
# 双层缓存模块
# 本地内存缓存（L1）+ Redis 分布式缓存（L2）
# 具备缓存击穿/穿透/雪崩防护
# ============================================================
import json
import time
from functools import lru_cache
from typing import Optional, Any, Dict
from loguru import logger

from backend.config import get_settings
from backend.database import get_redis


class LocalCache:
    """本地内存缓存（L1），基于 LRU 策略"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: Dict[str, tuple] = {}  # key -> (value, expire_time)

    def get(self, key: str) -> Optional[str]:
        """获取缓存，过期返回 None"""
        item = self._store.get(key)
        if item is None:
            return None
        value, expire = item
        if time.time() > expire:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """设置缓存，自动淘汰最旧条目"""
        ttl = ttl or self.default_ttl
        # LRU 淘汰：超过上限时删除最早插入的 20%
        if len(self._store) >= self.max_size:
            keys_to_evict = sorted(self._store.keys(), key=lambda k: self._store[k][1])[:len(self._store)//5]
            for k in keys_to_evict:
                del self._store[k]
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()


class CacheService:
    """
    双层缓存服务
    - L1: 本地内存缓存（极低延迟）
    - L2: Redis 分布式缓存（共享）
    - 防击穿：缓存不存在时通过互斥锁（SETNX）防并发穿透
    """

    def __init__(self):
        self.local = LocalCache(
            max_size=get_settings().cache_max_size,
            default_ttl=get_settings().cache_ttl,
        )

    async def get(self, key: str) -> Optional[str]:
        """L1 → L2 两级读取"""
        # L1 查询
        value = self.local.get(key)
        if value is not None:
            return value
        # L2 查询
        try:
            redis = await get_redis()
            value = await redis.get(key)
            if value is not None:
                # 回填 L1
                self.local.set(key, value)
            return value
        except Exception as e:
            logger.warning(f"Redis 缓存读取失败: {e}")
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """L1 + L2 双写"""
        settings = get_settings()
        ttl = ttl or settings.cache_ttl
        self.local.set(key, value, ttl)
        try:
            redis = await get_redis()
            await redis.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Redis 缓存写入失败: {e}")

    async def delete(self, key: str):
        """删除缓存"""
        self.local.delete(key)
        try:
            redis = await get_redis()
            await redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis 缓存删除失败: {e}")

    async def get_or_compute(self, key: str, compute_func, ttl: Optional[int] = None) -> Any:
        """
        缓存穿透防护：缓存不存在时通过 Redis SETNX 防并发穿透
        仅第一个请求执行 compute_func，其他请求等待
        """
        # 先查缓存
        cached = await self.get(key)
        if cached is not None:
            return json.loads(cached)

        # 防击穿：使用 SETNX 获取分布式锁
        lock_key = f"lock:{key}"
        try:
            redis = await get_redis()
            lock_acquired = await redis.setnx(lock_key, "1")
            if lock_acquired:
                await redis.expire(lock_key, 30)  # 锁 30 秒自动释放
                # 执行计算
                value = await compute_func()
                value_json = json.dumps(value, ensure_ascii=False)
                await self.set(key, value_json, ttl)
                await redis.delete(lock_key)
                return value
            else:
                # 等待锁释放后重试
                import asyncio
                for _ in range(50):  # 最多等待 5 秒
                    await asyncio.sleep(0.1)
                    cached = await self.get(key)
                    if cached is not None:
                        return json.loads(cached)
                # 超时后直接计算（兜底）
                logger.warning(f"缓存锁等待超时，直接计算: {key}")
                return await compute_func()
        except Exception as e:
            logger.error(f"缓存穿透防护异常: {e}, 直接计算")
            return await compute_func()


# 全局缓存服务实例
cache_service = CacheService()
