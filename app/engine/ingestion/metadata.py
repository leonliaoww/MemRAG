"""元数据绑定器 — 为每个 chunk 注入丰富的结构化元数据。

绑定的元数据字段：
    content_type    — 内容类型：text / table / code_block / image_description / image_ocr
    page_number     — 所在页码（PDF）
    heading_path    — 章节标题路径（如 "第三章 > 3.2 系统架构"）
    source_file     — 源文件名
    file_type       — 文件类型（pdf/markdown/docx）
    chunk_index     — chunk 在文档中的序号
    quality_score   — 清洗器给出的质量分
    has_table       — 是否包含表格内容
    has_code        — 是否包含代码块

这些元数据在 Chroma 中作为 metadata 字段存储，
支持检索时的过滤查询（如仅搜索特定章节、仅搜索正文等）。
"""

from __future__ import annotations

from langchain_core.documents import Document as LCDocument

from app.core.logging_config import get_logger
from app.engine.ingestion.parser import ElementType, ParsedDocument, ParsedElement

logger = get_logger(__name__)


def bind_metadata_from_parser(
    chunks: list[LCDocument],
    parsed_doc: ParsedDocument,
) -> list[LCDocument]:
    """将 parser 输出的结构信息绑定到每个 chunk 的 metadata。

    绑定策略：
        - 根据 chunk 内容在原始元素中的位置匹配标题路径
        - 将元素类型转换为 content_type
        - 继承页码和文件级元数据

    Args:
        chunks: 切分后的 chunk 列表。
        parsed_doc: parser 输出的 ParsedDocument。

    Returns:
        带完整元数据的 chunk 列表（原地修改）。
    """
    if not parsed_doc.elements:
        logger.debug("元数据绑定跳过", reason="parsed_doc 无元素")
        return chunks

    for chunk in chunks:
        content = chunk.page_content

        # 文件级元数据
        chunk.metadata["source_file"] = parsed_doc.file_path
        chunk.metadata["file_type"] = parsed_doc.file_type
        chunk.metadata["doc_metadata"] = parsed_doc.metadata

        # 查找最佳匹配的元素
        best_element = _find_matching_element(content, parsed_doc.elements)

        if best_element:
            chunk.metadata["content_type"] = best_element.type.value
            chunk.metadata["page_number"] = best_element.page_number
            chunk.metadata["heading_path"] = " > ".join(best_element.heading_path) if best_element.heading_path else ""

            # 内容特征标记
            chunk.metadata["has_table"] = best_element.type == ElementType.TABLE
            chunk.metadata["has_code"] = best_element.type == ElementType.CODE_BLOCK
            chunk.metadata["is_heading"] = best_element.type == ElementType.HEADING
        else:
            # 无法匹配时设定默认值
            chunk.metadata.setdefault("content_type", "text")
            chunk.metadata.setdefault("page_number", None)
            chunk.metadata.setdefault("heading_path", "")
            chunk.metadata.setdefault("has_table", False)
            chunk.metadata.setdefault("has_code", False)
            chunk.metadata.setdefault("is_heading", False)

    logger.debug("元数据绑定完成", chunks=len(chunks))
    return chunks


def _find_matching_element(content: str, elements: list[ParsedElement]) -> ParsedElement | None:
    """在元素列表中查找与 chunk 内容最匹配的元素。

    匹配策略（按优先级）：
        1. 精确包含匹配：元素内容完整包含在 chunk 中
        2. 前缀匹配：chunk 以元素内容开头
        3. 高重叠匹配：元素与 chunk 的 Jaccard 相似度最高
    """
    best_element: ParsedElement | None = None
    best_score = 0.0

    for el in elements:
        if not el.content:
            continue

        score = _overlap_score(content, el.content)

        if score > best_score:
            best_score = score
            best_element = el

    # 只有重叠度超过阈值才算匹配
    return best_element if best_score > 0.3 else None


def _overlap_score(chunk_text: str, element_text: str) -> float:
    """计算 chunk 文本与元素文本的重叠分数。

    使用简单的字符级 Jaccard 相似度（以 3-gram 为单位）。
    """
    def ngrams(text: str, n: int = 3) -> set:
        t = text.lower()
        return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}

    chunk_ngrams = ngrams(chunk_text)
    elem_ngrams = ngrams(element_text)

    if not chunk_ngrams or not elem_ngrams:
        return 0.0

    intersection = chunk_ngrams & elem_ngrams
    union = chunk_ngrams | elem_ngrams

    return len(intersection) / len(union) if union else 0.0


def bind_ocr_vlm_metadata(
    chunks: list[LCDocument],
    page_number: int | None = None,
    heading_path: str = "",
    source_file: str = "",
    file_type: str = "",
) -> list[LCDocument]:
    """为 OCR 和 VLM 产生的 chunk 绑定元数据。

    这些 chunk 的 content_type 已经是 image_ocr 或 image_description，
    此处补充文件级和位置元数据。
    """
    for chunk in chunks:
        chunk.metadata["source_file"] = source_file
        chunk.metadata["file_type"] = file_type
        chunk.metadata["page_number"] = page_number
        chunk.metadata["heading_path"] = heading_path
        chunk.metadata["has_table"] = False
        chunk.metadata["has_code"] = False
        chunk.metadata["is_heading"] = False

    return chunks
