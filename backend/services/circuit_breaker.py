# ============================================================
# 熔断器模块
# 为向量库查询、LLM 调用等外部依赖提供熔断、降级与重试能力
# ============================================================
import time
import asyncio
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.exceptions import CircuitBreakerOpenError


class CircuitState(Enum):
    CLOSED = "closed"       # 正常工作
    OPEN = "open"           # 熔断打开
    HALF_OPEN = "half_open" # 半开（试探恢复）


class CircuitBreaker:
    """
    熔断器实现
    - 连续 failure_threshold 次失败 → 熔断打开
    - 经过 recovery_timeout 秒 → 半开
    - 半开状态下请求成功 → 关闭；失败 → 再次打开
    """

    def __init__(self, name: str = "default", failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """在熔断保护下执行函数"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info(f"熔断器[{self.name}] 半开试探")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(f"熔断器[{self.name}] 已打开")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"熔断器[{self.name}] 已打开 (连续{self.failure_count}次失败)")
            raise e

        # 成功处理
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"熔断器[{self.name}] 恢复关闭")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        return result

    def __call__(self, func):
        """装饰器用法"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper


# ---------- 全局熔断器实例 ----------
vector_cb = CircuitBreaker(name="chromadb", failure_threshold=3, recovery_timeout=30.0)
llm_cb = CircuitBreaker(name="llm", failure_threshold=3, recovery_timeout=30.0)
embed_cb = CircuitBreaker(name="embedding", failure_threshold=3, recovery_timeout=30.0)


# ---------- 重试装饰器 ----------
def with_retry(max_attempts: int = 3):
    """带指数退避的重试装饰器"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, CircuitBreakerOpenError)),
        before_sleep=lambda retry_state: logger.warning(
            f"重试第{retry_state.attempt_number}次: {retry_state.outcome.exception()}"
        ),
    )
