"""文档管理 API — 上传、查询、删除文档。

端点：
    POST   /api/v1/documents/upload   — 上传并摄取文档（异步处理）
    GET    /api/v1/documents           — 列出所有文档元数据
    GET    /api/v1/documents/{id}      — 查询单个文档详情
    DELETE /api/v1/documents/{id}      — 删除文档及其向量数据

注意：MVP 阶段使用内存字典存储文档元数据，
      生产环境应替换为 PostgreSQL + SQLAlchemy。
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Path as FPath, UploadFile

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentUploadResponse,
)
from app.services.ingestion import ingest_document

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# 内存文档注册表 — MVP 阶段临时方案
# 生产环境替换为 PostgreSQL（使用 app/models/document.py 中的 ORM 模型）
_documents: dict[str, dict] = {}


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str | None = None,
) -> DocumentUploadResponse:
    """上传并异步摄取一个文档到知识库。

    处理流程：
        1. 校验文件扩展名和大小
        2. 保存到临时文件
        3. 调用摄取管道（加载→切分→嵌入→入库）
        4. 清理临时文件
        5. 返回文档 ID 和初始状态

    状态码 202：文档已接受，正在后台处理。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 校验文件扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'。支持的格式: {settings.SUPPORTED_EXTENSIONS}",
        )

    # 校验文件大小
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超出 {settings.MAX_UPLOAD_SIZE_MB} MB 限制",
        )

    doc_id = str(uuid.uuid4())

    # 写入临时文件（保留原始扩展名以供加载器识别）
    suffix = ext if ext else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # 注册文档元数据
    _documents[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "file_type": ext.lstrip("."),
        "file_size_bytes": len(content),
        "chunk_count": 0,
        "status": "processing",
        "error_message": None,
    }

    # 执行摄取管道（生产环境应提交到 Celery 任务队列）
    try:
        result = await ingest_document(
            file_path=tmp_path,
            collection_name=collection_name,
        )
        _documents[doc_id]["status"] = "completed"
        _documents[doc_id]["chunk_count"] = result.chunks
        logger.info("上传处理完成", doc_id=doc_id, chunks=result.chunks)
    except Exception as exc:
        _documents[doc_id]["status"] = "failed"
        _documents[doc_id]["error_message"] = str(exc)
        logger.error("上传处理失败", doc_id=doc_id, error=str(exc))
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return DocumentUploadResponse(
        id=uuid.UUID(doc_id),
        filename=file.filename,
        file_type=ext.lstrip("."),
        status=DocumentStatus(_documents[doc_id]["status"]),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """列出所有已上传的文档。"""
    items = [
        DocumentResponse(
            id=uuid.UUID(doc["id"]),
            knowledge_base_id=uuid.uuid4(),  # 占位（生产环境从 DB 读取）
            filename=doc["filename"],
            file_type=doc["file_type"],
            file_size_bytes=doc["file_size_bytes"],
            chunk_count=doc["chunk_count"],
            status=DocumentStatus(doc["status"]),
            error_message=doc.get("error_message"),
            created_at=None,  # type: ignore[arg-type]
        )
        for doc in _documents.values()
    ]
    return DocumentListResponse(total=len(items), items=items)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str = FPath(..., description="文档 UUID"),
) -> DocumentResponse:
    """查询单个文档的元数据和状态。"""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档未找到")

    return DocumentResponse(
        id=uuid.UUID(doc["id"]),
        knowledge_base_id=uuid.uuid4(),
        filename=doc["filename"],
        file_type=doc["file_type"],
        file_size_bytes=doc["file_size_bytes"],
        chunk_count=doc["chunk_count"],
        status=DocumentStatus(doc["status"]),
        error_message=doc.get("error_message"),
        created_at=None,  # type: ignore[arg-type]
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str = FPath(..., description="文档 UUID"),
) -> None:
    """删除文档及其在 Chroma 中的所有 chunk 向量。

    注意：MVP 阶段仅从内存注册表删除。
          生产环境需同步删除 Chroma 中的向量数据。
    """
    doc = _documents.pop(document_id, None)
    if not doc:
        raise HTTPException(status_code=404, detail="文档未找到")
    logger.info("文档已删除", doc_id=document_id)
