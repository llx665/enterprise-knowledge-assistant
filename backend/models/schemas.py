# ============================================================
# Pydantic Schema 定义
# 所有请求/响应数据模型统一在此定义，确保类型安全
# ============================================================
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ---------- 权限相关 ----------
class UserRole(str, Enum):
    visitor = "visitor"
    admin = "admin"


class UserInfo(BaseModel):
    username: str = "visitor"
    role: UserRole = UserRole.visitor


class LoginRequest(BaseModel):
    username: str = Field("admin", description="用户名")
    password: str = Field("admin123", description="密码")


class LoginResponse(BaseModel):
    code: int = 200
    message: str = "登录成功"
    data: Optional[Dict[str, Any]] = None


# ---------- 文档相关 ----------
class ChunkInfo(BaseModel):
    """文档分块信息"""
    chunk_id: str = ""
    content: str = ""
    parent_id: str = ""
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    metadata: Dict[str, Any] = {}


class DocumentInfo(BaseModel):
    """文档元数据"""
    doc_id: str = ""
    filename: str = ""
    file_type: str = ""
    file_size: int = 0
    chunk_count: int = 0
    status: str = "pending"  # pending / processing / completed / failed
    upload_time: str = ""
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: List[DocumentInfo] = []


class DeleteRequest(BaseModel):
    doc_ids: List[str] = Field(..., description="待删除文档ID列表")


# ---------- 检索相关 ----------
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    session_id: str = Field("default", description="会话ID，用于记忆")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数")
    user_agent: str = "web"


class CitationSource(BaseModel):
    """引用来源"""
    doc_id: str = ""
    filename: str = ""
    content: str = ""
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    score: float = 0.0


class SearchResponse(BaseModel):
    """SSE 流式响应的最终聚合结构"""
    answer: str = ""
    citations: List[CitationSource] = []
    need_retrieval: bool = True
    self_check_passed: bool = True


# ---------- 日志相关 ----------
class LogRecord(BaseModel):
    id: str = ""
    timestamp: str = ""
    action: str = ""  # upload / search / delete / rebuild
    user: str = ""
    detail: str = ""
    duration_ms: float = 0.0
    query: Optional[str] = None
    recall_count: int = 0


class LogQueryParams(BaseModel):
    action: Optional[str] = None
    user: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class LogListResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: List[LogRecord] = []
    total: int = 0
    page: int = 1


class LogStats(BaseModel):
    """日志统计"""
    avg_duration_ms: float = 0.0
    total_searches: int = 0
    top_queries: List[Dict[str, Any]] = []


# ---------- 索引相关 ----------
class IndexVersion(BaseModel):
    version_id: str = ""
    created_at: str = ""
    doc_count: int = 0
    chunk_count: int = 0
    description: str = ""


class IndexInfoResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Dict[str, Any] = {}


# ---------- 通用响应 ----------
class ApiResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


# ---------- 评测相关 ----------
class EvalResult(BaseModel):
    faithfulness: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    report_path: str = ""
    sample_count: int = 0


class EvalResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[EvalResult] = None
