"""OCR + VLM 视觉理解 — 提取文档中图片的文字和语义信息。

双管线处理：
    1. OCR（离线）— 使用 unstructured 内置 OCR 提取图片中的嵌入文字
    2. VLM（云端）— 使用 DashScope 多模态 API（qwen-vl）生成图片语义描述

处理后的结果作为特殊 chunk 存入向量库：
    - OCR 文字 chunk：与正文 chunk 同等对待，可被检索
    - VLM 描述 chunk：标注为 image_description 类型，包含原图上下文

配置：
    OCR_ENABLED=true   → 启用 OCR（离线，需安装 tesseract）
    VLM_ENABLED=true   → 启用 VLM（需 DASHSCOPE_API_KEY）
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.ingestion.parser import ElementType, ParsedDocument, ParsedElement

logger = get_logger(__name__)


@dataclass
class VisionResult:
    """单张图片的 OCR + VLM 处理结果。"""
    ocr_text: str = ""
    """OCR 提取的图片内文字（可能为空）。"""
    vlm_description: str = ""
    """VLM 生成的图片语义描述（可能为空）。"""
    source_element: ParsedElement | None = None
    """来源图片元素（包含位置、页码等元数据）。"""
    error: str | None = None
    """处理过程中的错误信息。"""


# ── OCR 处理 ─────────────────────────────────────────

def _ocr_image(image_base64: str, image_mime: str) -> str:
    """对单张 base64 编码的图片执行 OCR。

    使用 unstructured 内置的 OCR 功能（底层依赖 tesseract）。
    需要系统安装 tesseract-ocr：
        macOS:   brew install tesseract tesseract-lang
        Ubuntu:  apt install tesseract-ocr tesseract-ocr-chi-sim
        Windows: https://github.com/UB-Mannheim/tesseract/wiki

    支持中英文混合识别。
    """
    try:
        # 将 base64 解码为字节
        image_bytes = base64.b64decode(image_base64)

        # 使用 unstructured 的 OCR 功能
        from unstructured.partition.image import partition_image

        elements = partition_image(
            file=image_bytes,
            # OCR 语言：中文简体 + 英文
            ocr_languages="chi_sim+eng",
            strategy="hi_res",
        )

        text_parts = [str(el).strip() for el in elements if str(el).strip()]
        return "\n".join(text_parts)

    except ImportError:
        logger.warning("OCR 不可用", reason="unstructured 或 tesseract 未安装")
        return ""
    except Exception as exc:
        logger.error("OCR 处理失败", error=str(exc))
        return ""


# ── VLM 处理 ─────────────────────────────────────────

_VLM_PROMPT = """请分析这张文档中的图片，提取以下信息：

1. 图片类型：这是图表（折线图/柱状图/饼图）、流程图、架构图、照片还是截图？
2. 关键内容：图片展示了什么信息？包含哪些关键数据和标签？
3. 与文档上下文的关联：这张图片可能在说明什么内容？

请用简洁的中文回答，3-5 句话即可。
如果图片中的文字已经被 OCR 提取，请在此基础上补充语义理解。"""


def _vlm_describe_image(
    image_base64: str,
    image_mime: str = "image/png",
    ocr_text: str = "",
) -> str:
    """使用 DashScope 多模态 API 对图片进行语义描述。

    Args:
        image_base64: 图片的 base64 编码。
        image_mime: 图片 MIME 类型。
        ocr_text: OCR 已提取的文字（作为附加上下文提供给 VLM）。

    Returns:
        VLM 生成的图片语义描述。如果 VLM 不可用则返回空字符串。
    """
    if not settings.DASHSCOPE_API_KEY:
        logger.warning("VLM 不可用", reason="DASHSCOPE_API_KEY 未设置")
        return ""

    try:
        import dashscope
        from dashscope import MultiModalConversation

        # 构建多模态消息
        context = ""
        if ocr_text:
            context = f"\n\n[OCR 已提取的图片文字]\n{ocr_text}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:{image_mime};base64,{image_base64}"},
                    {"text": _VLM_PROMPT + context},
                ],
            }
        ]

        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            api_key=settings.DASHSCOPE_API_KEY.get_secret_value(),
        )

        if response.status_code == 200:
            output = response.output
            if output and output.choices:
                return output.choices[0].message.content[0]["text"]
            return ""
        else:
            logger.error("VLM 调用失败", code=response.status_code, message=response.message)
            return ""

    except ImportError:
        logger.warning("VLM 不可用", reason="dashscope 未安装。pip install dashscope")
        return ""
    except Exception as exc:
        logger.error("VLM 处理失败", error=str(exc))
        return ""


# ── 统一入口 ────────────────────────────────────────

@dataclass
class DocumentVisionResult:
    """整个文档的 OCR + VLM 处理结果集合。"""
    results: list[VisionResult] = field(default_factory=list)
    """每个图片元素的处理结果。"""

    @property
    def ocr_chunks(self) -> list[LCDocument]:
        """将 OCR 结果转换为可索引的 LangChain Document 列表。

        每个图片的 OCR 文字作为一个独立 chunk，
        携带图片来源的完整元数据（页码、标题路径等）。
        """
        docs: list[LCDocument] = []
        for i, r in enumerate(self.results):
            if r.ocr_text.strip():
                el = r.source_element
                docs.append(LCDocument(
                    page_content=f"[图片 {i + 1} OCR 文字]\n{r.ocr_text}",
                    metadata={
                        "content_type": "image_ocr",
                        "page_number": el.page_number if el else None,
                        "heading_path": " > ".join(el.heading_path) if el and el.heading_path else "",
                        "chunk_index": -1,
                    },
                ))
        return docs

    @property
    def vlm_chunks(self) -> list[LCDocument]:
        """将 VLM 描述转换为可索引的 LangChain Document 列表。

        VLM 描述标注为 image_description 类型，与正文 chunk 区分开，
        便于检索时按需获取图片语义信息。
        """
        docs: list[LCDocument] = []
        for i, r in enumerate(self.results):
            if r.vlm_description.strip():
                el = r.source_element
                docs.append(LCDocument(
                    page_content=f"[图片 {i + 1} 语义描述]\n{r.vlm_description}",
                    metadata={
                        "content_type": "image_description",
                        "page_number": el.page_number if el else None,
                        "heading_path": " > ".join(el.heading_path) if el and el.heading_path else "",
                        "chunk_index": -1,
                    },
                ))
        return docs


def process_document_images(
    parsed_doc: ParsedDocument,
    enable_ocr: bool | None = None,
    enable_vlm: bool | None = None,
) -> DocumentVisionResult:
    """处理文档中的所有图片元素。

    根据配置决定启用哪些管线：
        - OCR：离线提取图片内的嵌入文字
        - VLM：云端语义理解图片内容

    建议处理顺序：先 OCR 提取文字 → 将 OCR 文字作为上下文传给 VLM，
    这样 VLM 可以在已有文字基础上做更准确的语义理解。

    Args:
        parsed_doc: parser.py 输出的 ParsedDocument。
        enable_ocr: 覆盖配置，强制开启/关闭 OCR。
        enable_vlm: 覆盖配置，强制开启/关闭 VLM。

    Returns:
        DocumentVisionResult 包含每个图片的处理结果和可索引 chunk。
    """
    do_ocr = enable_ocr if enable_ocr is not None else settings.OCR_ENABLED
    do_vlm = enable_vlm if enable_vlm is not None else settings.VLM_ENABLED

    images = parsed_doc.image_elements

    if not images:
        logger.debug("文档无图片，跳过视觉处理", file=parsed_doc.file_path)
        return DocumentVisionResult()

    logger.info(
        "视觉处理开始",
        file=parsed_doc.file_path,
        image_count=len(images),
        ocr=do_ocr,
        vlm=do_vlm,
    )

    results: list[VisionResult] = []

    for i, img_element in enumerate(images):
        result = VisionResult(source_element=img_element)

        if not img_element.image_base64:
            result.error = "图片无 base64 数据"
            results.append(result)
            continue

        # 第1步：OCR 提取文字
        if do_ocr:
            logger.debug("OCR 处理中", image_index=i, page=img_element.page_number)
            result.ocr_text = _ocr_image(
                img_element.image_base64,
                img_element.image_mime or "image/png",
            )
            if result.ocr_text:
                logger.debug("OCR 完成", image_index=i, text_len=len(result.ocr_text))

        # 第2步：VLM 语义理解（传入 OCR 结果作为上下文）
        if do_vlm:
            logger.debug("VLM 处理中", image_index=i, page=img_element.page_number)
            result.vlm_description = _vlm_describe_image(
                img_element.image_base64,
                img_element.image_mime or "image/png",
                ocr_text=result.ocr_text,
            )
            if result.vlm_description:
                logger.debug("VLM 完成", image_index=i, desc_len=len(result.vlm_description))

        results.append(result)

    doc_result = DocumentVisionResult(results=results)
    logger.info(
        "视觉处理完成",
        file=parsed_doc.file_path,
        ocr_chunks=len(doc_result.ocr_chunks),
        vlm_chunks=len(doc_result.vlm_chunks),
    )
    return doc_result
