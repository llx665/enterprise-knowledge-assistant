# ============================================================
# 日志模块
# 记录上传、检索、删除等可审计操作，并提供日志查询接口
# 底层使用 SQLite 存储日志记录
# ============================================================
import json
import sqlite3
import time
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from loguru import logger

from backend.models.schemas import LogRecord, LogQueryParams

DB_PATH = "./data/logs/operation_logs.db"


def _get_db():
    """获取数据库连接（线程安全）"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """初始化日志表"""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS operation_logs (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            user TEXT NOT NULL DEFAULT 'visitor',
            detail TEXT DEFAULT '',
            duration_ms REAL DEFAULT 0.0,
            query TEXT,
            recall_count INTEGER DEFAULT 0
        )
    """)
    # 创建索引
    conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON operation_logs(action)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON operation_logs(timestamp)")
    conn.commit()
    conn.close()


# 启动时初始化
_init_db()


class LoggerService:
    """操作日志服务"""

    @staticmethod
    def log(action: str, user: str = "visitor", detail: str = "",
            duration_ms: float = 0.0, query: Optional[str] = None,
            recall_count: int = 0):
        """记录一条操作日志"""
        try:
            conn = _get_db()
            record = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action,
                "user": user,
                "detail": detail[:500],
                "duration_ms": round(duration_ms, 1),
                "query": (query or "")[:200],
                "recall_count": recall_count,
            }
            conn.execute(
                """INSERT INTO operation_logs
                   (id, timestamp, action, user, detail, duration_ms, query, recall_count)
                   VALUES (:id, :timestamp, :action, :user, :detail, :duration_ms, :query, :recall_count)""",
                record,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"日志记录失败: {e}")

    @staticmethod
    def query(params: LogQueryParams) -> tuple:
        """查询日志，返回 (records, total_count)"""
        conn = _get_db()
        conditions = []
        values = {}

        if params.action:
            conditions.append("action = :action")
            values["action"] = params.action
        if params.user:
            conditions.append("user = :user")
            values["user"] = params.user

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        offset = (params.page - 1) * params.page_size

        # 计数
        total = conn.execute(
            f"SELECT COUNT(*) FROM operation_logs WHERE {where_clause}", values
        ).fetchone()[0]

        # 分页查询
        rows = conn.execute(
            f"SELECT * FROM operation_logs WHERE {where_clause} ORDER BY timestamp DESC LIMIT :limit OFFSET :offset",
            {**values, "limit": params.page_size, "offset": offset},
        ).fetchall()

        records = []
        for row in rows:
            records.append(LogRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                action=row["action"],
                user=row["user"],
                detail=row["detail"],
                duration_ms=row["duration_ms"],
                query=row["query"] or "",
                recall_count=row["recall_count"],
            ))

        conn.close()
        return records, total

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """获取统计信息"""
        conn = _get_db()

        # 总检索数
        total_searches = conn.execute(
            "SELECT COUNT(*) FROM operation_logs WHERE action = 'search'"
        ).fetchone()[0]

        # 平均耗时
        avg_duration = conn.execute(
            "SELECT AVG(duration_ms) FROM operation_logs WHERE action = 'search' AND duration_ms > 0"
        ).fetchone()[0] or 0.0

        # 高频提问 TOP10
        top_queries = conn.execute(
            """SELECT query, COUNT(*) as cnt FROM operation_logs
               WHERE action = 'search' AND query != '' AND query IS NOT NULL
               GROUP BY query ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()

        top_list = [{"query": row["query"], "count": row["cnt"]} for row in top_queries]

        conn.close()

        return {
            "avg_duration_ms": round(avg_duration, 1),
            "total_searches": total_searches,
            "top_queries": top_list,
        }


logger_service = LoggerService()
