"""文档加载器工厂 — 将任意支持的文件格式加载为 LangChain Document 对象。

支持格式：PDF、DOCX、TXT、Markdown、HTML、CSV。

使用 Unstructured 库作为主解析引擎，LangChain 社区加载器作为备选。
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
    UnstructuredPDFLoader,
)

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 文件扩展名 → 加载器类的注册表
# 注册新格式时在此处添加映射即可，无需修改业务代码
LOADER_REGISTRY: dict[str, type] = {
    ".pdf": UnstructuredPDFLoader,
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

    # 将文件级别的元数据注入到每一个 Document 中
    # 这些元数据会贯穿整个管道，最终可用于过滤和引用追踪
    file_meta = {
        "source": str(path),            # 源文件绝对路径
        "filename": path.name,          # 文件名
        "file_type": ext.lstrip("."),   # 文件类型（不含点）
        "file_size_bytes": path.stat().st_size,  # 文件大小
    }
    for doc in docs:
        doc.metadata.update(file_meta)

    logger.info("文档加载完成", path=str(path), chunks=len(docs))
    return docs
