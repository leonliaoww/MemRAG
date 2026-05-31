"""文档加载器工厂 — 将任意支持的文件格式加载为 LangChain Document 对象。

支持格式：PDF、DOCX、TXT、Markdown、HTML、CSV。

PDF 使用 PyMuPDF (fitz) 解析，无需外部依赖。
其他格式使用 LangChain 社区加载器。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PyMuPDFLoader:
    """使用 PyMuPDF (fitz) 加载 PDF 文档。

    纯 Python 实现，无需 poppler 等外部依赖。
    支持提取文本、保留页面结构。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[LCDocument]:
        """加载 PDF 并返回 Document 列表（每页一个）。"""
        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF 未安装。请运行: pip install pymupdf"
            ) from exc

        docs: list[LCDocument] = []
        with fitz.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    doc = LCDocument(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "total_pages": len(pdf),
                        },
                    )
                    docs.append(doc)

        return docs


LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".csv": CSVLoader,
}


def load_document(file_path: str | Path) -> list[LCDocument]:
    """从磁盘加载一个文档并返回 LangChain Document 对象列表。

    内部逻辑：
        1. 校验文件是否存在
        2. 根据扩展名查找对应的加载器
        3. 调用加载器解析文档
        4. 为每个解析出的页面/段落注入文件级元数据

    Args:
        file_path: 文档文件的路径（字符串或 Path 对象）。

    Returns:
        LangChain Document 对象列表（每页/每段一个）。

    Raises:
        ValueError: 文件扩展名不在支持列表中。
        FileNotFoundError: 文件不存在。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {file_path}")

    ext = path.suffix.lower()
    loader_cls = LOADER_REGISTRY.get(ext)

    if loader_cls is None:
        supported = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(
            f"不支持的文件类型 '{ext}'。支持的格式: {supported}"
        )

    logger.info("加载文档中", path=str(path), loader=loader_cls.__name__)

    loader = loader_cls(str(path))
    docs = loader.load()

    file_meta = {
        "source": str(path),
        "filename": path.name,
        "file_type": ext.lstrip("."),
        "file_size_bytes": path.stat().st_size,
    }
    for doc in docs:
        doc.metadata.update(file_meta)

    logger.info("文档加载完成", path=str(path), chunks=len(docs))
    return docs
