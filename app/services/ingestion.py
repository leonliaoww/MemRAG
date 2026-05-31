"""文档摄取编排器。

将 加载→切分→嵌入→存储 串联为一个完整的异步管道。
支持单个文档和批量目录两种模式。
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging_config import get_logger
from app.engine.ingestion.embedder import embed_documents
from app.engine.ingestion.loader import load_document
from app.engine.ingestion.splitter import split_documents
from app.engine.retrieval.vector_store import add_documents

logger = get_logger(__name__)


class IngestionResult:
    """单次摄取运行的产出统计。

    包含：
        - 源文件路径
        - 原始页面/段落数
        - 切分后的 chunk 数
        - Chroma 中分配的唯一 ID 列表
    """

    def __init__(
        self,
        file_path: str,
        raw_pages: int,
        chunks: int,
        chroma_ids: list[str],
    ) -> None:
        self.file_path = file_path
        self.raw_pages = raw_pages
        self.chunks = chunks
        self.chroma_ids = chroma_ids


async def ingest_document(
    file_path: str | Path,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionResult:
    """对单个文档运行完整摄取管道。

    流程：加载文件 → 切分 chunk → 生成嵌入 → 存入 Chroma

    Args:
        file_path: 文档文件路径。
        collection_name: 目标 Chroma collection（不传使用默认）。
        chunk_size: 覆盖默认 chunk 大小。
        chunk_overlap: 覆盖默认重叠大小。

    Returns:
        IngestionResult 包含处理统计和 Chroma ID 列表。
    """
    path = Path(file_path)

    logger.info("摄取开始", file=path.name)

    # 第1步：加载文档
    raw_docs = load_document(str(path))
    logger.info("文档已加载", pages=len(raw_docs))

    # 第2步：切分 chunk
    chunks = split_documents(raw_docs, chunk_size, chunk_overlap)
    logger.info("文档已切分", chunks=len(chunks))

    # 第3步和第4步：嵌入 + 存储（Chroma 内部调用 embedding function）
    ids = await add_documents(chunks, collection_name)

    logger.info(
        "摄取完成",
        file=path.name,
        pages=len(raw_docs),
        chunks=len(chunks),
        ids=len(ids),
    )

    return IngestionResult(
        file_path=str(path),
        raw_pages=len(raw_docs),
        chunks=len(chunks),
        chroma_ids=ids,
    )


async def ingest_directory(
    directory: str | Path,
    collection_name: str | None = None,
    glob_pattern: str = "*.*",
) -> list[IngestionResult]:
    """批量摄取目录下的所有匹配文件。

    每个文件独立处理，某个文件失败不影响其他文件。

    Args:
        directory: 要扫描的目录路径。
        collection_name: 目标 Chroma collection。
        glob_pattern: 文件匹配模式（如 "*.pdf" 或 "*.*"）。

    Returns:
        成功处理的每个文件的结果列表。
    """
    dir_path = Path(directory)
    files = sorted(dir_path.glob(glob_pattern))

    results: list[IngestionResult] = []
    for file_path in files:
        if file_path.is_file():
            try:
                result = await ingest_document(file_path, collection_name)
                results.append(result)
            except Exception as exc:
                # 单个文件失败不中断整个批处理
                logger.error("文件摄取失败", file=str(file_path), error=str(exc))

    logger.info(
        "目录摄取完成",
        total=len(results),
        failed=len(files) - len(results),
    )
    return results
