# ============================================================
# ???????
# ============================================================
import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger
from backend.models.schemas import (
    ApiResponse, DocumentInfo, DocumentListResponse,
    DeleteRequest, UserRole,
)
from backend.services.auth_service import auth_service
from backend.services.logger_service import logger_service
from backend.config import get_settings

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Management"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    import sqlite3
    conn = sqlite3.connect("./data/logs/operation_logs.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT detail FROM operation_logs WHERE action = 'upload' ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    documents = []
    seen = set()
    for row in rows:
        detail = row["detail"]
        if "upload" in detail:
            parts = detail.replace("upload file: ", "").split(" (")
            filename = parts[0].strip() if parts else "unknown"
            if filename not in seen:
                seen.add(filename)
                documents.append(DocumentInfo(
                    doc_id=f"doc_{len(seen)}", filename=filename,
                    file_type=os.path.splitext(filename)[1].lower(),
                    status="completed",
                ))
    conn.close()
    docs_dir = "./data/documents"
    if os.path.exists(docs_dir):
        for fname in os.listdir(docs_dir):
            if fname not in seen:
                ext = os.path.splitext(fname)[1].lower()
                fp = os.path.join(docs_dir, fname)
                documents.append(DocumentInfo(
                    doc_id=os.path.splitext(fname)[0], filename=fname,
                    file_type=ext, file_size=os.path.getsize(fp),
                    status="completed",
                ))
    return DocumentListResponse(data=documents)


@router.post("/documents/delete", response_model=ApiResponse)
async def delete_documents(req: DeleteRequest, request: Request):
    user = auth_service.get_user_from_request(request)
    if user.role != UserRole.admin:
        return JSONResponse(status_code=403, content={"code": 403, "message": "Admin permission required", "data": None})
    try:
        import redis as sync_redis
        from rq import Queue
        settings = get_settings()
        sync_conn = sync_redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, password=settings.redis_password or None)
        q = Queue("default", connection=sync_conn)
        from backend.tasks.document_processing import delete_documents_task
        q.enqueue(delete_documents_task, req.doc_ids)
    except Exception:
        from backend.tasks.document_processing import delete_documents_task
        delete_documents_task(req.doc_ids)
    logger_service.log(action="delete", user=user.username, detail=f"Delete docs: {req.doc_ids}")
    return ApiResponse(message=f"Delete task submitted for {len(req.doc_ids)} documents")


from fastapi.responses import FileResponse


@router.get("/documents/view", response_model=None)
async def view_document(filename: str):
    """查看文档内容（纯文本/Markdown 直接显示，PDF 返回下载）"""
    import os
    docs_dir = "./data/documents"
    # Search all files in the directory
    for fname in os.listdir(docs_dir):
        if fname == filename or fname.endswith(filename):
            filepath = os.path.join(docs_dir, fname)
            if os.path.isfile(filepath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".md", ".txt", ".html"):
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    return ApiResponse(data={"filename": fname, "content": content, "file_type": ext})
                else:
                    # PDF etc: return as file download
                    return FileResponse(filepath, filename=fname)
    return ApiResponse(code=404, message="文档未找到")


@router.get("/stats", response_model=ApiResponse)
async def knowledge_stats():
    docs_dir = "./data/documents"
    doc_count = 0
    total_size = 0
    if os.path.exists(docs_dir):
        for fname in os.listdir(docs_dir):
            fp = os.path.join(docs_dir, fname)
            if os.path.isfile(fp):
                doc_count += 1
                total_size += os.path.getsize(fp)
    return ApiResponse(data={"doc_count": doc_count, "total_size_bytes": total_size,
                             "total_size_mb": round(total_size / 1024 / 1024, 2)})
