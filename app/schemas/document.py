"""文档相关 Pydantic Schema — 上传、查询、知识库管理。"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """文档处理状态枚举。"""
    PENDING = "pending"        # 等待处理
    PROCESSING = "processing"  # 正在处理中
    COMPLETED = "completed"    # 处理完成
    FAILED = "failed"          # 处理失败


class DocumentUploadResponse(BaseModel):
    """文档提交后立即返回的响应 — 状态初始为 pending。"""
    id: uuid.UUID
    filename: str
    file_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    message: str = "文档已提交，正在排队处理"


class DocumentResponse(BaseModel):
    """文档完整元数据 — GET 请求返回的详细结构。"""
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: DocumentStatus
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}  # 支持从 ORM 模型直接转换


class DocumentListResponse(BaseModel):
    """文档分页列表响应。"""
    total: int
    items: list[DocumentResponse]


class KnowledgeBaseCreate(BaseModel):
    """创建知识库的请求体。"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None


class KnowledgeBaseResponse(BaseModel):
    """知识库元数据响应。"""
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None = None
    chroma_collection: str     # 对应的 Chroma collection 名称
    document_count: int = 0    # 知识库中的文档数量
    created_at: datetime

    model_config = {"from_attributes": True}
