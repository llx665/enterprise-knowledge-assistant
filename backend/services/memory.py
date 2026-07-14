# ============================================================
# 会话记忆模块
# 包含短期会话摘要（SLiding窗口）和长期用户偏好记忆（持久化JSON）
# ============================================================
import json
import time
import os
from typing import List, Dict, Optional, Any
from collections import deque
from loguru import logger

from backend.config import get_settings


class ShortTermMemory:
    """
    短期会话记忆（滑动窗口）
    存储最近 N 轮用户-助手对话摘要，用于上下文理解
    """

    def __init__(self, max_rounds: int = 10, ttl: int = 1800):
        self.max_rounds = max_rounds
        self.ttl = ttl
        # 格式: {session_id: deque([(timestamp, role, content), ...])}
        self._sessions: Dict[str, deque] = {}

    def _ensure_session(self, session_id: str) -> deque:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_rounds)
        return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """添加对话记录"""
        queue = self._ensure_session(session_id)
        queue.append((time.time(), role, content))
        self._gc(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取会话历史（格式化为 LLM 可用的消息列表）"""
        queue = self._ensure_session(session_id)
        self._gc(session_id)
        messages = []
        for _, role, content in queue:
            messages.append({"role": role, "content": content})
        return messages

    def get_summary(self, session_id: str) -> str:
        """生成简短会话摘要"""
        messages = self.get_history(session_id)
        if not messages:
            return ""
        # 取最后 3 轮对话的缩写摘要
        recent = messages[-6:]  # 最多 3 轮（一问一答为一轮）
        summary_parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            summary_parts.append(f"{role}: {content}")
        return "\n".join(summary_parts)

    def _gc(self, session_id: str):
        """清理过期记录"""
        queue = self._sessions.get(session_id)
        if not queue:
            return
        now = time.time()
        while queue and (now - queue[0][0] > self.ttl):
            queue.popleft()

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)


class LongTermMemory:
    """
    长期用户偏好记忆（持久化 JSON 文件）
    记录用户的检索偏好、热词、关注领域等，用于个性化检索权重调整
    """

    def __init__(self, storage_path: str = "./data/memory.json"):
        self.storage_path = storage_path
        self._data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        """从磁盘加载长期记忆"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"长期记忆加载失败: {e}")
        return {}

    def _save(self):
        """持久化至磁盘"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"长期记忆保存失败: {e}")

    def record_search(self, user_id: str, query: str, top_keywords: List[str]):
        """记录用户检索行为，更新偏好"""
        if user_id not in self._data:
            self._data[user_id] = {
                "queries": [],
                "keywords": {},
                "interests": {},
                "total_searches": 0,
            }
        profile = self._data[user_id]

        # 记录最近查询（最多 100 条）
        profile["queries"].append({"query": query, "time": time.time()})
        if len(profile["queries"]) > 100:
            profile["queries"] = profile["queries"][-100:]

        # 更新关键词权重
        for kw in top_keywords[:10]:
            profile["keywords"][kw] = profile["keywords"].get(kw, 0) + 1

        profile["total_searches"] += 1
        self._save()

    def get_keyword_weights(self, user_id: str) -> Dict[str, float]:
        """获取用户关键词偏好权重"""
        profile = self._data.get(user_id)
        if not profile:
            return {}
        total = sum(profile["keywords"].values())
        if total == 0:
            return {}
        return {k: v / total for k, v in profile["keywords"].items()}

    def get_search_history(self, user_id: str, limit: int = 10) -> List[str]:
        """获取用户最近检索历史"""
        profile = self._data.get(user_id)
        if not profile:
            return []
        return [q["query"] for q in profile["queries"][-limit:]]

    def get_total_searches(self, user_id: str) -> int:
        profile = self._data.get(user_id)
        return profile["total_searches"] if profile else 0


# 全局单例
settings = get_settings()
short_memory = ShortTermMemory(max_rounds=10, ttl=settings.short_memory_ttl)
long_memory = LongTermMemory(storage_path=settings.long_memory_path)
