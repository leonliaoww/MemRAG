"""文档智能切分策略。

使用 LangChain 的 RecursiveCharacterTextSplitter 进行递归切分，
支持中英文混合场景。切分时保留 chunk 重叠以保持语义连贯性。

未来扩展方向：
    - 语义切分（基于 embedding 相似度的断点检测）
    - 句子窗口切分（检索时返回周围上下文）
    - 代码感知切分（基于 AST 的代码块切分）
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建配置好的 RecursiveCharacterTextSplitter 实例。

    分隔符按优先级从高到低排列：
        段落 → 换行 → 英文句号 → 中文句号 → 空格 → 字符

    Args:
        chunk_size: chunk 大小（字符数），不传则使用配置默认值。
        chunk_overlap: 相邻 chunk 重叠字符数。
        separators: 自定义分隔符列表。

    Returns:
        配置完成的切分器实例。
    """
    if separators is None:
        # 中英文混合分隔符：优先在自然断点处切分
        separators = ["\n\n", "\n", ". ", "。", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=separators,
        keep_separator=True,        # 保留分隔符，维持原文可读性
        add_start_index=True,       # 记录 chunk 在原文中的起始位置（用于引用定位）
    )


def split_documents(
    documents: list[LCDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[LCDocument]:
    """将文档列表切分为固定大小的 chunk。

    每个 chunk 都会被自动标注 chunk_index 元数据，
    用于后续的引用追踪和源定位。

    Args:
        documents: 从加载器获得的原始文档列表。
        chunk_size: 覆盖默认的 chunk 大小。
        chunk_overlap: 覆盖默认的重叠大小。

    Returns:
        切分后的 Document chunk 列表，每个都带有 chunk_index 元数据。
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # 为每个 chunk 标注索引号（引用追踪的关键元数据）
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "文档切分完成",
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=chunk_size or settings.CHUNK_SIZE,
    )
    return chunks
