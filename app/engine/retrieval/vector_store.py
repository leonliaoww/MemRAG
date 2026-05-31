"""Chroma 向量库封装 — collection 管理和文档操作。

通过 LangChain 的 Chroma 集成实现，支持：
- 多知识库隔离（每个知识库对应一个 collection）
- 异步文档添加和删除
- 相似度搜索（带分数）

Chroma 部署模式：
    - 开发环境：Docker 容器（docker compose up chroma）
    - 生产环境：独立 Chroma 服务集群
"""

from __future__ import annotations

import uuid
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.ingestion.embedder import get_embeddings

logger = get_logger(__name__)

# 全局 Chroma HTTP 客户端（单例）
_chroma_client: chromadb.ClientAPI | None = None
# 向量库缓存 — 按 collection 名称缓存 LangChain Chroma 实例
_vector_stores: dict[str, Chroma] = {}


def _get_chroma_client() -> chromadb.ClientAPI:
    """获取 Chroma HTTP 客户端单例。

    首次调用时创建 HTTP 连接，后续调用复用。
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(
            "Chroma 客户端初始化完成",
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
    return _chroma_client


def get_vector_store(
    collection_name: str | None = None,
) -> Chroma:
    """获取指定 collection 的 LangChain Chroma 向量库实例。

    同一 collection 只创建一次，后续调用从缓存获取。

    Args:
        collection_name: Chroma collection 名称。默认使用配置中的 collection。

    Returns:
        绑定到 OpenAI embeddings 的 LangChain Chroma 实例。
    """
    name = collection_name or settings.CHROMA_COLLECTION_NAME

    if name not in _vector_stores:
        client = _get_chroma_client()
        embeddings = get_embeddings()

        _vector_stores[name] = Chroma(
            client=client,
            collection_name=name,
            embedding_function=embeddings,
        )
        logger.info("向量库已创建", collection=name)

    return _vector_stores[name]


def create_collection(collection_name: str) -> Chroma:
    """创建新的 Chroma collection（用于新知识库的初始化）。

    如果同名 collection 已存在，先删除再创建（幂等操作）。

    Args:
        collection_name: 新 collection 的唯一名称。

    Returns:
        新创建的 LangChain Chroma 实例。
    """
    client = _get_chroma_client()
    embeddings = get_embeddings()

    # 幂等处理：先尝试删除同名 collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection 不存在，忽略错误

    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    _vector_stores[collection_name] = store
    logger.info("Collection 创建成功", name=collection_name)
    return store


def delete_collection(collection_name: str) -> bool:
    """删除一个 Chroma collection 及其所有向量数据。

    此操作不可逆，会删除 collection 中的所有文档嵌入。

    Returns:
        删除成功返回 True，collection 不存在返回 False。
    """
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
        _vector_stores.pop(collection_name, None)
        logger.info("Collection 已删除", name=collection_name)
        return True
    except Exception:
        return False


async def add_documents(
    documents: list[LCDocument],
    collection_name: str | None = None,
) -> list[str]:
    """将文档添加到向量库（自动嵌入 + 存储）。

    每个文档分配一个 UUID 作为 Chroma 中的唯一 ID，
    便于后续精确删除和引用追踪。

    Args:
        documents: 已切分并带元数据的 LangChain Document 列表。
        collection_name: 目标 collection 名称。

    Returns:
        分配的 Chroma 文档 ID 列表（与输入一一对应）。
    """
    store = get_vector_store(collection_name)
    ids = [str(uuid.uuid4()) for _ in documents]

    await store.aadd_documents(documents, ids=ids)
    logger.info("文档已存入 Chroma", count=len(documents), collection=collection_name)
    return ids


async def search_similar(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    filter_metadata: dict | None = None,
) -> list[LCDocument]:
    """向量相似度搜索 — 基于余弦距离。

    返回的每个文档的 metadata 中会自动附加 `relevance_score` 字段。

    Args:
        query: 自然语言查询字符串。
        top_k: 返回结果数量，默认使用配置值。
        collection_name: 搜索的目标 collection。
        filter_metadata: 可选的元数据过滤条件（如 {"file_type": "pdf"}）。

    Returns:
        按相似度降序排列的 Document 列表。
    """
    store = get_vector_store(collection_name)
    k = top_k or settings.RETRIEVAL_TOP_K

    # similarity_search_with_relevance_scores 返回 (doc, score) 元组
    results = await store.asimilarity_search_with_relevance_scores(
        query, k=k, filter=filter_metadata
    )

    # 将相似度分数附加到每个文档的元数据中
    docs: list[LCDocument] = []
    for doc, score in results:
        doc.metadata["relevance_score"] = score
        docs.append(doc)

    logger.debug("向量检索完成", query=query[:80], results=len(docs))
    return docs


async def delete_documents(
    ids: list[str],
    collection_name: str | None = None,
) -> None:
    """按 ID 从 Chroma 中批量删除文档。

    Args:
        ids: 要删除的 Chroma 文档 ID 列表。
        collection_name: 目标 collection。
    """
    store = get_vector_store(collection_name)
    await store.adelete(ids)
    logger.info("Chroma 文档已删除", count=len(ids))


def collection_stats(collection_name: str | None = None) -> dict:
    """获取 collection 的基本统计信息。

    主要用于健康检查和监控。

    Returns:
        包含 name 和 count（文档总数）的字典。
    """
    client = _get_chroma_client()
    name = collection_name or settings.CHROMA_COLLECTION_NAME
    try:
        coll = client.get_collection(name)
        return {"name": name, "count": coll.count()}
    except Exception:
        return {"name": name, "count": 0}
