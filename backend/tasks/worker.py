# ============================================================
# RQ Worker 入口
# 处理异步任务：文档解析、向量入库、RAG 评测等
# ============================================================
import os
import sys
# 确保导入路径正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import redis
from rq import Worker, Queue, Connection
from loguru import logger

from backend.config import get_settings


def start_worker():
    """启动 RQ Worker"""
    settings = get_settings()

    redis_conn = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
    )

    with Connection(redis_conn):
        worker = Worker(["default", "high", "low"])
        logger.info("RQ Worker 已启动，监听队列: default, high, low")
        worker.work()


if __name__ == "__main__":
    start_worker()
