# ============================================================
# 文档上传路由
# 多格式批量上传、上传进度展示
# ============================================================
import os
import uuid
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from backend.models.schemas import ApiResponse, UserInfo, UserRole
from backend.services.auth_service import auth_service
from backend.services.logger_service import logger_service

router = APIRouter(prefix="/api/upload", tags=["文档上传"])


@router.post("", response_model=ApiResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="待上传文件列表"),
):
    """批量上传文档，返回任务ID（异步处理）"""
    # 权限校验
    user = auth_service.get_user_from_request(request)
    if user.role != UserRole.admin:
        return JSONResponse(status_code=403, content={"code": 403, "message": "需要管理员权限", "data": None})

    if not files:
        return ApiResponse(code=400, message="请选择文件")

    upload_dir = "./data/documents"
    os.makedirs(upload_dir, exist_ok=True)

    uploaded = []
    errors = []

    for file in files:
        if not file.filename:
            continue

        # 校验扩展名
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}:
            errors.append(f"{file.filename}: 不支持的文件格式")
            continue

        try:
            # 保存文件
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            safe_filename = f"{doc_id}{ext}"
            file_path = os.path.join(upload_dir, safe_filename)

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # 提交异步处理任务
            # 直接异步处理（不走 RQ 队列，避免阻塞）
            from backend.services.parser import parser
            from backend.services.vector_store import vector_store
            from backend.services.bm25_retriever import bm25_retriever

            # 解析文档
            parent_chunks = parser.parse(file_path, file.filename)
            if parent_chunks:
                all_chunks = []
                for parent in parent_chunks:
                    all_chunks.extend(parent["chunks"])

                # 向量入库
                added = await vector_store.add_documents(doc_id, all_chunks)

                # 更新 BM25 索引
                bm25_chunks = [
                    {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
                    for c in all_chunks
                ]
                bm25_retriever.load_or_build(bm25_chunks)

                logger.info(f"文档处理完成: {file.filename} | {added} 块")

            uploaded.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "file_size": len(content),
                "task_id": doc_id,
                "status": "processing",
            })

            logger_service.log(
                action="upload", user=user.username,
                detail=f"上传文件: {file.filename} ({len(content)} bytes)",
            )

        except Exception as e:
            logger.error(f"上传失败 {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")

    return ApiResponse(data={"uploaded": uploaded, "errors": errors})


@router.get("/extensions", response_model=ApiResponse)
async def get_supported_extensions():
    """获取支持的文件格式列表"""
    return ApiResponse(data={
        "extensions": [".pdf", ".docx", ".html", ".htm", ".md", ".txt"],
        "max_size_mb": 50,
    })
