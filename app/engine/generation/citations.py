"""引用追踪 — 从生成的答案中提取和匹配来源引用。

引用格式约定：
    答案中出现的 [filename.pdf] 格式被视为来源引用。
    该模块负责：
        1. 从答案文本中提取所有引用标签
        2. 将引用标签与检索到的文档进行模糊匹配
        3. 生成结构化的 SourceCitation 对象

设计说明：
    当前采用"文本解析"方式（正则匹配 [文件名]），
    优点是对 LLM 无侵入，缺点是依赖于 LLM 遵循格式指令。
    未来可升级为 function calling 方式，让 LLM 显式返回引用列表。
"""

from __future__ import annotations

import re

from langchain_core.documents import Document as LCDocument

from app.schemas.query import SourceCitation

# 匹配 [xxx] 格式的引用标签
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def extract_citation_refs(text: str) -> list[str]:
    """从答案字符串中提取所有引用标签。

    例如：
        输入 "营收210万 [Q4_report.pdf]，同比增长15% [annual.pdf]"
        返回 ["Q4_report.pdf", "annual.pdf"]

    自动去重，保留首次出现顺序。

    Args:
        text: LLM 生成的带引用标签的答案文本。

    Returns:
        去重后的引用文件名列表。
    """
    matches = CITATION_PATTERN.findall(text)
    # 去重但保持顺序
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def build_citations(
    answer: str,
    retrieved_docs: list[LCDocument],
) -> list[SourceCitation]:
    """将答案中的引用标签与检索文档进行匹配。

    匹配策略：模糊匹配 — 引用标签包含在文件名中（或反过来），
    不区分大小写。

    Args:
        answer: 带引用标签的答案文本。
        retrieved_docs: 作为上下文提供给 LLM 的检索文档列表。

    Returns:
        SourceCitation 对象列表（仅包含匹配到的引用）。
    """
    cited_refs = extract_citation_refs(answer)
    if not cited_refs:
        return []

    citations: list[SourceCitation] = []
    for doc in retrieved_docs:
        filename = doc.metadata.get("filename", "unknown")
        for ref in cited_refs:
            # 模糊匹配：引用标签是文件名的子串或反之
            if ref.lower() in filename.lower() or filename.lower() in ref.lower():
                citations.append(
                    SourceCitation(
                        document_id=doc.metadata.get("id", ""),
                        filename=filename,
                        chunk_index=doc.metadata.get("chunk_index", 0),
                        content_preview=doc.page_content[:300],
                        relevance_score=doc.metadata.get("cross_encoder_score")
                        or doc.metadata.get("relevance_score"),
                    )
                )
                break  # 一个文档只匹配一次

    return citations


def format_context(documents: list[LCDocument], max_chars: int = 6000) -> str:
    """将检索到的文档格式化为 LLM 可读的上下文字符串。

    每个文档包含：
        - 编号（Document N）
        - 来源文件名
        - 文档正文（如超出长度限制则截断）

    Args:
        documents: 检索到的文档 chunk 列表。
        max_chars: 上下文最大字符数（超出部分截断最后一个文档）。

    Returns:
        格式化的上下文字符串，文档之间用 "---" 分隔。
    """
    parts: list[str] = []
    total = 0

    for i, doc in enumerate(documents, start=1):
        filename = doc.metadata.get("filename", "unknown")
        header = f"[文档 {i}] 来源: {filename}\n"
        body = doc.page_content.strip()
        block = header + body

        # 如果加上当前文档会超出限制，截断最后一个文档
        if total + len(block) > max_chars and parts:
            remaining = max_chars - total
            if remaining > 100:  # 剩余空间足够容纳有意义的内容
                block_truncated = header + body[:remaining] + "\n... [已截断]"
                parts.append(block_truncated)
            break

        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)
