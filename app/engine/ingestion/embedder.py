"""嵌入服务 — 封装 OpenAI embeddings API，提供批量嵌入和查询嵌入。

使用单例模式复用连接，避免每次请求都创建新的 HTTP 客户端。

OpenAI text-embedding-3 系列支持 dimensions 参数，
可在精度和成本之间灵活调节（默认 1536 维）。
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 单例嵌入客户端 — 全局复用 HTTP 连接池
_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """获取 OpenAEmbeddings 单例实例。

    首次调用时创建，后续调用直接返回缓存的实例。
    线程安全（OpenAIEmbeddings 内部使用 httpx 连接池）。
    """
    global _embeddings
    if _embeddings is None:
        if settings.OPENAI_API_KEY is None:
            raise RuntimeError(
                "OPENAI_API_KEY 未设置。请在 .env 文件中配置 OPENAI_API_KEY"
            )
        _embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            dimensions=1536,  # text-embedding-3 支持维度压缩
        )
        logger.info(
            "嵌入模型初始化完成",
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
    return _embeddings


async def embed_documents(documents: list[LCDocument]) -> list[list[float]]:
    """为文档列表批量生成嵌入向量。

    内部调用 OpenAI 的批量 embedding API，
    自动处理速率限制和重试。

    Args:
        documents: 待嵌入的文档 chunk 列表。

    Returns:
        嵌入向量列表（每个文档对应一个 1536 维向量）。
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
        1536 维嵌入向量。
    """
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(query)
    return vector
