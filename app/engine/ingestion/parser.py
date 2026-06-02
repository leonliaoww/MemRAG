"""高级文档解析器 — 处理复杂 PDF 和 Markdown 文档。

功能：
    PDF：使用 unstructured 的 hi_res 策略，自动检测表格、多栏布局、嵌入图片
    Markdown：保留标题层级、代码块、表格等语义结构

输出统一 ParsedDocument 结构：
    - 文本元素（正文段落、标题、表格、代码块）
    - 图片元素（嵌入图片的位置和 base64 数据）
    - 结构元数据（章节层级、页码）

设计原则：
    - 尽可能保留文档原始结构信息
    - 表格和代码块作为独立元素，不与被周围文本混淆
    - 图片区域标识出来，后续交由 OCR/VLM 管线处理
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ── 数据结构 ────────────────────────────────────────

class ElementType(str, Enum):
    """文档元素类型。"""
    HEADING = "heading"       # 标题
    PARAGRAPH = "paragraph"   # 普通段落
    TABLE = "table"           # 表格
    CODE_BLOCK = "code_block" # 代码块
    LIST_ITEM = "list_item"   # 列表项
    IMAGE = "image"           # 嵌入图片
    CAPTION = "caption"       # 图片/表格标题


@dataclass
class ParsedElement:
    """文档中解析出的单个元素。

    属性：
        type: 元素类型（段落/表格/图片/标题等）
        content: 文本内容（图片类型时为空字符串）
        page_number: 所在页码（PDF），Markdown 为 None
        heading_path: 该元素所属的标题层级路径，如 ["第一章", "1.1 概述"]
        image_base64: 图片的 base64 编码数据（仅 IMAGE 类型）
        image_mime: 图片 MIME 类型（仅 IMAGE 类型）
        bbox: 元素在页面中的边界框坐标（x1,y1,x2,y2），无坐标时为 None
    """
    type: ElementType
    content: str = ""
    page_number: int | None = None
    heading_path: list[str] = field(default_factory=list)
    image_base64: str | None = None
    image_mime: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedDocument:
    """解析后的文档结构。

    属性：
        file_path: 源文件路径
        file_type: 文件类型（pdf / markdown / docx 等）
        elements: 解析出的所有元素列表
        metadata: 文档级别的元数据（作者、标题、创建时间等）
    """
    file_path: str
    file_type: str
    elements: list[ParsedElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def text_elements(self) -> list[ParsedElement]:
        """所有非图片元素（正文、表格、标题等）。"""
        return [e for e in self.elements if e.type != ElementType.IMAGE]

    @property
    def image_elements(self) -> list[ParsedElement]:
        """所有图片元素。"""
        return [e for e in self.elements if e.type == ElementType.IMAGE]


# ── PDF 解析 ────────────────────────────────────────

def _parse_pdf_hi_res(file_path: str) -> ParsedDocument:
    """使用 unstructured 的 hi_res 策略解析复杂 PDF。

    hi_res 策略额外调用：
        - 表格检测模型（识别表格区域）
        - 布局分析模型（检测多栏、图片区域）
        - OCR（对扫描件提取文字）
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError(
            "unstructured[all-docs] 未安装。请运行: pip install 'unstructured[all-docs]'"
        )

    logger.info("PDF hi_res 解析中", file=file_path)

    # 使用 hi_res 策略解析 PDF
    raw_elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        # 提取图片（转换为 base64 供 VLM 使用）
        extract_images_in_pdf=True,
        # 提取表格结构
        infer_table_structure=True,
        # 语言检测
        languages=["chi_sim", "eng"],
    )

    path = Path(file_path)
    doc = ParsedDocument(file_path=file_path, file_type="pdf")

    # 当前标题栈（跟踪章节层级）
    heading_stack: list[str] = []

    for el in raw_elements:
        el_type = str(el.category).lower() if hasattr(el, "category") else ""

        # 确定元素类型
        element_type = _map_unstructured_type(el_type, el.to_dict() if hasattr(el, "to_dict") else {})

        # 获取页码
        page_number = el.metadata.page_number if hasattr(el.metadata, "page_number") else None

        # 获取文本内容
        content = str(el) if hasattr(el, "__str__") else ""

        # 处理标题层级
        if element_type == ElementType.HEADING:
            _update_heading_stack(heading_stack, content)

        # 处理图片元素
        image_base64 = None
        image_mime = None
        if element_type == ElementType.IMAGE:
            el_dict = el.to_dict() if hasattr(el, "to_dict") else {}
            image_base64 = el_dict.get("metadata", {}).get("image_base64")
            image_mime = el_dict.get("metadata", {}).get("image_mime_type")

        element = ParsedElement(
            type=element_type,
            content=content.strip(),
            page_number=page_number,
            heading_path=list(heading_stack),
            image_base64=image_base64,
            image_mime=image_mime,
        )
        doc.elements.append(element)

    logger.info(
        "PDF 解析完成",
        file=file_path,
        text_elements=len(doc.text_elements),
        image_elements=len(doc.image_elements),
    )
    return doc


def _parse_pdf_fast(file_path: str) -> ParsedDocument:
    """快速 PDF 解析（不使用 hi_res，速度快但表格/图片检测弱）。

    作为 hi_res 的轻量备选，适合纯文本 PDF。
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError("unstructured[all-docs] 未安装")

    logger.info("PDF 快速解析中", file=file_path)

    raw_elements = partition_pdf(
        filename=file_path,
        strategy="fast",
    )

    path = Path(file_path)
    doc = ParsedDocument(file_path=file_path, file_type="pdf")
    heading_stack: list[str] = []

    for el in raw_elements:
        el_type = str(el.category).lower() if hasattr(el, "category") else ""
        element_type = _map_unstructured_type(el_type, {})
        page_number = el.metadata.page_number if hasattr(el.metadata, "page_number") else None
        content = str(el) if hasattr(el, "__str__") else ""

        if element_type == ElementType.HEADING:
            _update_heading_stack(heading_stack, content)

        doc.elements.append(ParsedElement(
            type=element_type,
            content=content.strip(),
            page_number=page_number,
            heading_path=list(heading_stack),
        ))

    logger.info("PDF 解析完成（快速模式）", file=file_path, elements=len(doc.elements))
    return doc


# ── Markdown 解析 ────────────────────────────────────

def _parse_markdown(file_path: str) -> ParsedDocument:
    """解析 Markdown 文档，保留标题层级、代码块和表格。

    处理策略：
        - # 标题 → HEADING 类型，跟踪层级
        - ```代码块``` → CODE_BLOCK 类型
        - | 表格 | → TABLE 类型
        - ![图片]() → IMAGE 类型，提取路径
        - 其余 → PARAGRAPH 类型
    """
    from unstructured.partition.md import partition_md

    logger.info("Markdown 解析中", file=file_path)

    raw_elements = partition_md(filename=file_path)

    doc = ParsedDocument(file_path=file_path, file_type="markdown")
    heading_stack: list[str] = []

    # 辅助：检测 Markdown 标题级别
    import re
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    for el in raw_elements:
        content = str(el).strip() if hasattr(el, "__str__") else ""
        el_type = str(el.category).lower() if hasattr(el, "category") else "paragraph"

        element_type = _map_unstructured_type(el_type, {})

        # 手动检测标题层级（unstructured 的标题检测不稳定）
        if element_type != ElementType.HEADING:
            m = heading_pattern.match(content)
            if m:
                element_type = ElementType.HEADING

        if element_type == ElementType.HEADING:
            _update_heading_stack(heading_stack, content.lstrip("#").strip())

        doc.elements.append(ParsedElement(
            type=element_type,
            content=content,
            heading_path=list(heading_stack),
        ))

    logger.info("Markdown 解析完成", file=file_path, elements=len(doc.elements))
    return doc


# ── 辅助函数 ────────────────────────────────────────

def _map_unstructured_type(unstructured_type: str, element_dict: dict) -> ElementType:
    """将 unstructured 的元素类型映射到统一的 ElementType。"""
    type_lower = unstructured_type.lower()

    if "table" in type_lower or "tablecell" in type_lower:
        return ElementType.TABLE
    if "header" in type_lower or "title" in type_lower or "heading" in type_lower:
        return ElementType.HEADING
    if "code" in type_lower:
        return ElementType.CODE_BLOCK
    if "list" in type_lower:
        return ElementType.LIST_ITEM
    if "image" in type_lower:
        return ElementType.IMAGE
    if "caption" in type_lower:
        return ElementType.CAPTION
    if "figure" in type_lower:
        return ElementType.IMAGE

    return ElementType.PARAGRAPH


def _update_heading_stack(stack: list[str], heading_text: str) -> None:
    """更新标题层级栈。

    简化策略：以标题文本长度判断层级（markdown # 数量）。
    生产环境应使用更精确的标题编号匹配。
    """
    # 清理标题文本
    clean = heading_text.lstrip("#").strip()
    if not clean:
        return

    # 简单策略：如果栈非空且当前标题看起来是同级的，替换最后一个
    # 这里做简化处理，生产环境可扩展为完整层级管理
    if stack and len(clean) < len(stack[-1]) * 2:
        stack.pop()
    stack.append(clean)
    # 限制栈深度，避免过长
    if len(stack) > 5:
        stack.pop(0)


# ── 统一入口 ────────────────────────────────────────

def parse_document(file_path: str | Path) -> ParsedDocument:
    """解析任意支持格式的文档，返回统一 ParsedDocument 结构。

    根据文件类型和配置自动选择最合适的解析策略：
        - PDF：默认 hi_res（支持表格/图片/多栏），配置可切换为 fast
        - Markdown：保留完整语义结构
        - 其他格式：回退到基础 loader

    Args:
        file_path: 文档路径。

    Returns:
        包含结构化元素列表的 ParsedDocument。
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _parse_pdf_fast(str(path))
    elif ext in (".md", ".markdown"):
        return _parse_markdown(str(path))
    else:
        # 其他格式回退到基础 loader
        from app.engine.ingestion.loader import load_document
        raw_docs = load_document(str(path))
        doc = ParsedDocument(file_path=str(path), file_type=ext.lstrip("."))

        for raw in raw_docs:
            content = raw.page_content.strip()
            if content:
                doc.elements.append(ParsedElement(
                    type=ElementType.PARAGRAPH,
                    content=content,
                ))

        logger.info("文档解析完成（基础模式）", file=path.name, elements=len(doc.elements))
        return doc
