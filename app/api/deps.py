"""FastAPI 依赖注入 — 为路由处理函数提供共享资源。

提供的依赖：
    - settings: 应用配置
    - get_store(): Chroma 向量库实例
"""

from __future__ import annotations

from app.core.config import settings
from app.engine.retrieval.vector_store import get_vector_store


def get_settings():
    """依赖注入：应用配置单例。"""
    return settings


def get_store(collection_name: str | None = None):
    """依赖注入：获取指定 collection 的 Chroma 向量库。

    Args:
        collection_name: 目标 collection 名称。None 时使用默认配置。

    Returns:
        LangChain Chroma 实例。
    """
    return get_vector_store(collection_name)
