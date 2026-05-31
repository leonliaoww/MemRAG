"""嵌入服务 — 多供应商嵌入模型封装。

支持供应商：
    - 阿里云 DashScope：text-embedding-v3（1024维，中英文优化）
    - OpenAI：text-embedding-3-small / text-embedding-3-large

通过 LLM_PROVIDER 配置选择供应商，运行时自动加载对应适配器。
使用单例模式复用连接。
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 单例嵌入客户端
_embeddings: Embeddings | None = None


def _create_dashscope_embeddings() -> Embeddings:
    """创建阿里云 DashScope 嵌入客户端。

    使用 langchain_community 的 DashScopeEmbeddings 封装。
    text-embedding-v3 默认输出 1024 维向量，支持中英文混合场景。
    """
    from langchain_community.embeddings import DashScopeEmbeddings

    if settings.DASHSCOPE_API_KEY is None:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未设置。请在 .env 中配置阿里云 DashScope API Key"
        )

    return DashScopeEmbeddings(
        model=settings.DASHSCOPE_EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY.get_secret_value(),
    )


def _create_openai_embeddings() -> Embeddings:
    """创建 OpenAI 嵌入客户端。"""
    from langchain_openai import OpenAIEmbeddings

    if settings.OPENAI_API_KEY is None:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置。请在 .env 中配置 OpenAI API Key"
        )

    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        dimensions=1536,
    )


def get_embeddings() -> Embeddings:
    """获取嵌入模型单例（根据 LLM_PROVIDER 自动选择供应商）。

    首次调用时创建实例，后续调用复用。
    """
    global _embeddings
    if _embeddings is None:
        if settings.LLM_PROVIDER == "dashscope":
            _embeddings = _create_dashscope_embeddings()
            logger.info(
                "嵌入模型已初始化",
                provider="dashscope",
                model=settings.DASHSCOPE_EMBEDDING_MODEL,
            )
        else:
            _embeddings = _create_openai_embeddings()
            logger.info(
                "嵌入模型已初始化",
                provider="openai",
                model=settings.OPENAI_EMBEDDING_MODEL,
            )
    return _embeddings


async def embed_documents(documents: list[LCDocument]) -> list[list[float]]:
    """为文档列表批量生成嵌入向量。

    内部调用供应商的批量 embedding API，自动处理速率限制和重试。

    Args:
        documents: 待嵌入的文档 chunk 列表。

    Returns:
        嵌入向量列表（每个文档对应一个向量，维度取决于供应商）。
    """
    embeddings = get_embeddings()
    texts = [doc.page_content for doc in documents]
    vectors = await embeddings.aembed_documents(texts)
    logger.info("文档嵌入完成", count=len(vectors))
    return vectors


async def embed_query(query: str) -> list[float]:
    """为单个查询字符串生成嵌入向量。

    用于检索阶段的查询向量化。

    Args:
        query: 用户查询字符串。

    Returns:
        嵌入向量。
    """
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(query)
    return vector
