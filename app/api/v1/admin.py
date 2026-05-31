"""管理 API — 健康检查、Collection 管理、系统信息。

端点：
    GET  /api/v1/admin/health         — 系统健康检查（含 Chroma 连通性）
    GET  /api/v1/admin/collections    — 列出所有 Chroma collection 及统计
    POST /api/v1/admin/collections/{name} — 创建新 collection（新知识库）
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.engine.retrieval.vector_store import collection_stats, create_collection
from app.schemas.common import HealthResponse

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """系统健康检查 — 验证 Chroma 和 Redis 连通性。

    返回状态：
        - "ok": 所有服务正常
        - "degraded": Chroma 不可用但应用仍运行

    用于：Kubernetes liveness/readiness probe、负载均衡器健康检查。
    """
    chroma_ok = True
    redis_ok = True

    # 检查 Chroma 连通性
    try:
        stats = collection_stats()
        chroma_ok = "count" in stats
    except Exception:
        chroma_ok = False

    # 检查 Redis 连通性（不可用不影响核心功能）
    try:
        import redis as _redis
        r = _redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
    except Exception:
        redis_ok = False

    return HealthResponse(
        status="ok" if chroma_ok else "degraded",
        version=settings.APP_VERSION,
        chroma_connected=chroma_ok,
        redis_connected=redis_ok,
    )


@router.get("/collections")
async def list_collections():
    """列出所有 Chroma collection 及其文档统计。

    返回格式：
        {"collections": [{"name": "...", "count": N}]}
    """
    try:
        stats = collection_stats()
        return {"collections": [stats]}
    except Exception as exc:
        return {"collections": [], "error": str(exc)}


@router.post("/collections/{name}", status_code=201)
async def create_new_collection(name: str):
    """创建新的 Chroma collection — 用于新建知识库。

    如果同名 collection 已存在，会被删除后重建（幂等操作）。

    Args:
        name: collection 名称（建议使用知识库 UUID 或唯一标识符）。
    """
    store = create_collection(name)
    stats = collection_stats(name)
    return {"name": name, "count": stats.get("count", 0)}
