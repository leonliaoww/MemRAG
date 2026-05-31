"""通用 Pydantic Schema — 健康检查、分页、错误响应。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应 — 用于负载均衡器和监控探针。"""
    status: str = "ok"
    version: str
    chroma_connected: bool   # Chroma 向量库连接状态
    redis_connected: bool    # Redis 连接状态


class ErrorResponse(BaseModel):
    """标准化错误响应格式 — 所有 HTTP 异常的统一输出结构。"""
    detail: str                      # 人类可读的错误描述
    error_code: str | None = None    # 机器可读的错误码（如 "DOC_NOT_FOUND"）
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """基于偏移量的分页参数。"""
    offset: int = Field(default=0, ge=0, description="偏移量")
    limit: int = Field(default=20, ge=1, le=100, description="每页数量")


class PaginatedResponse(BaseModel):
    """通用分页响应包装器 — 所有分页接口的统一输出格式。"""
    total: int      # 总记录数
    offset: int     # 当前偏移量
    limit: int      # 每页数量
    items: list     # 当前页数据
