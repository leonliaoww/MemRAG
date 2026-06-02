"""文档摄取编排器 — 7 阶段全链路管道。

管道流程：
    parse → vision → split → clean → metadata → embed → index(BM25+Chroma)

每个阶段独立可跳过，通过配置控制（OCR_ENABLED、VLM_ENABLED 等）。
支持单文件和批量目录两种模式。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.ingestion.cleaner import clean_chunks
from app.engine.ingestion.metadata import bind_metadata_from_parser, bind_ocr_vlm_metadata
from app.engine.ingestion.parser import ParsedDocument, parse_document
from app.engine.ingestion.splitter import split_documents
from app.engine.ingestion.vision import process_document_images
from app.engine.retrieval.bm25_index import get_bm25_index
from app.engine.retrieval.vector_store import add_documents

logger = get_logger(__name__)


class IngestionResult:
    """单次摄取运行的完整统计。

    包含每个阶段的产出数量，便于监控和调优。
    """

    def __init__(
        self,
        file_path: str,
        raw_elements: int = 0,
        image_elements: int = 0,
        ocr_chunks: int = 0,
        vlm_chunks: int = 0,
        text_chunks: int = 0,
        cleaned_chunks: int = 0,
        chroma_ids: list[str] | None = None,
    ) -> None:
        self.file_path = file_path
        self.raw_elements = raw_elements        # parser 输出的元素总数
        self.image_elements = image_elements    # 检测到的图片数量
        self.ocr_chunks = ocr_chunks            # OCR 产生的 chunk 数
        self.vlm_chunks = vlm_chunks            # VLM 产生的 chunk 数
        self.text_chunks = text_chunks          # 切分后的文本 chunk 数
        self.cleaned_chunks = cleaned_chunks    # 清洗后的 chunk 数
        self.chroma_ids = chroma_ids or []      # Chroma 分配的 ID 列表


async def ingest_document(
    file_path: str | Path,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionResult:
    """对单个文档运行完整的 7 阶段摄取管道。

    阶段说明：
        1. parse      — 高级解析（PDF hi_res / Markdown 结构 / 其他格式兜底）
        2. vision     — OCR 提取图片文字 + VLM 生成图片语义描述（可配置关闭）
        3. split      — 文本元素切分为固定大小 chunk
        4. clean      — 去噪、过滤低质量 chunk、去重
        5. metadata   — 绑定页码、标题层级、内容类型等结构化元数据
        6. embed      — 生成稠密向量（嵌入模型）→ 存入 Chroma
        7. index      — 更新 BM25 持久化索引

    Args:
        file_path: 文档文件路径。
        collection_name: 目标 Chroma collection。
        chunk_size: 覆盖默认 chunk 大小。
        chunk_overlap: 覆盖默认重叠大小。

    Returns:
        IngestionResult 包含每个阶段的产出统计。
    """
    path = Path(file_path)

    logger.info("摄取开始（7阶段管道）", file=path.name)
    result = IngestionResult(file_path=str(path))

    # ── 阶段 1：解析 ──────────────────────────────────
    parsed_doc = parse_document(str(path))
    result.raw_elements = len(parsed_doc.elements)
    result.image_elements = len(parsed_doc.image_elements)
    logger.info("阶段1 解析完成", elements=result.raw_elements, images=result.image_elements)

    # ── 阶段 2：视觉（OCR + VLM）───────────────────────
    vision_result = process_document_images(parsed_doc)
    ocr_chunks = vision_result.ocr_chunks
    vlm_chunks = vision_result.vlm_chunks
    result.ocr_chunks = len(ocr_chunks)
    result.vlm_chunks = len(vlm_chunks)
    logger.info("阶段2 视觉完成", ocr_chunks=result.ocr_chunks, vlm_chunks=result.vlm_chunks)

    # ── 阶段 3：切分 ──────────────────────────────────
    # 将文本元素转为 LangChain Document 列表
    from langchain_core.documents import Document as LCDocument

    text_docs: list[LCDocument] = []
    for el in parsed_doc.text_elements:
        if el.content.strip():
            text_docs.append(LCDocument(
                page_content=el.content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": parsed_doc.file_type,
                    "page_number": el.page_number,
                    "heading_path": " > ".join(el.heading_path) if el.heading_path else "",
                    "element_type": el.type.value,
                },
            ))

    chunks = split_documents(text_docs, chunk_size, chunk_overlap)
    logger.info("阶段3 切分完成", chunks=len(chunks))

    # ── 阶段 4：清洗 ──────────────────────────────────
    chunks = clean_chunks(chunks)
    result.text_chunks = len(chunks)
    logger.info("阶段4 清洗完成", chunks=len(chunks))

    # ── 阶段 5：元数据绑定 ────────────────────────────
    chunks = bind_metadata_from_parser(chunks, parsed_doc)

    # OCR 和 VLM chunk 也绑定元数据
    page_num = parsed_doc.elements[0].page_number if parsed_doc.elements else None
    heading_path = " > ".join(parsed_doc.elements[0].heading_path) if parsed_doc.elements and parsed_doc.elements[0].heading_path else ""

    ocr_chunks = bind_ocr_vlm_metadata(
        ocr_chunks,
        page_number=page_num,
        heading_path=heading_path,
        source_file=str(path),
        file_type=parsed_doc.file_type,
    )
    vlm_chunks = bind_ocr_vlm_metadata(
        vlm_chunks,
        page_number=page_num,
        heading_path=heading_path,
        source_file=str(path),
        file_type=parsed_doc.file_type,
    )

    # 合并所有 chunk：文本 + OCR + VLM
    all_chunks = chunks + ocr_chunks + vlm_chunks
    result.cleaned_chunks = len(all_chunks)
    logger.info("阶段5 元数据绑定完成", total_chunks=len(all_chunks))

    # ── 阶段 6：嵌入 + 向量存储 ────────────────────────
    chroma_ids = await add_documents(all_chunks, collection_name)
    result.chroma_ids = chroma_ids
    logger.info("阶段6 嵌入存储完成", ids=len(chroma_ids))

    # ── 阶段 7：BM25 索引更新 ──────────────────────────
    bm25 = get_bm25_index(collection_name)
    # 为每个 chunk 设置 ID（与 Chroma 一致，用于后续精确删除）
    for chunk, cid in zip(all_chunks, chroma_ids):
        chunk.metadata["id"] = cid
    bm25.add_documents(all_chunks)
    bm25.save()
    logger.info("阶段7 BM25索引更新完成", total_docs=bm25.doc_count)

    logger.info(
        "摄取完成（全链路）",
        file=path.name,
        elements=result.raw_elements,
        images=result.image_elements,
        chunks=result.cleaned_chunks,
    )
    return result


async def ingest_directory(
    directory: str | Path,
    collection_name: str | None = None,
    glob_pattern: str = "*.*",
) -> list[IngestionResult]:
    """批量摄取目录下的所有匹配文件。

    每个文件独立处理，单个文件失败不影响其他文件。
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
                logger.error("文件摄取失败", file=str(file_path), error=str(exc))

    logger.info(
        "目录摄取完成",
        total=len(results),
        failed=len(files) - len(results),
    )
    return results
