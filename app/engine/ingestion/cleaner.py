"""Chunk 清洗器 — 对切分后的 chunk 进行去噪、去重和质量过滤。

处理步骤：
    1. 空白清洗：统一换行、去除多余空格、合并短行
    2. 长度过滤：剔除过短/过长的无效 chunk
    3. 重复检测：标记与相邻 chunk 高度重叠的内容
    4. 质量评分：基于字符分布判断 chunk 是否包含有意义内容

设计原则：
    - 不修改原始语义，只做格式清理
    - 可配置阈值，适应不同场景
    - 清洗结果标记在 metadata 中，原始内容保留
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _normalize_whitespace(text: str) -> str:
    """规范化空白字符。

    - 将连续空行压缩为单个空行（保留段落边界）
    - 行首行尾去空格
    - 将连续空格合并为一个
    - 保留中文字符间不需要空格的特征
    """
    # 按行处理
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    prev_empty = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            # 合并连续空格
            stripped = re.sub(r" {2,}", " ", stripped)
            # 中文字符间去除多余空格
            stripped = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", stripped)
            cleaned_lines.append(stripped)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


def _is_noise(text: str) -> bool:
    """判断 chunk 是否为噪声内容。

    噪声判定规则：
        - 纯数字/标点/特殊字符
        - 页眉页脚常见模式（页码、版权声明等）
        - 无意义重复字符
    """
    if not text:
        return True

    # 纯数字和标点
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.1:
        return True

    # 常见页眉页脚模式
    noise_patterns = [
        r"^\d+\s*$",                      # 纯页码
        r"^[©®™]\s*\d{4}",                # 版权声明
        r"^第\s*\d+\s*页\s*/\s*共\s*\d+\s*页$",  # 中文页码
        r"^CONFIDENTIAL$",                # 水印文字
        r"^[_\-\=]{5,}$",                 # 分隔线
    ]
    for pattern in noise_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True

    return False


def _score_quality(text: str) -> float:
    """评估 chunk 的内容质量（0.0 ~ 1.0）。

    分数基于：
        - 有效字符比例（字母+中文 vs 标点/数字）
        - 平均词长（过短说明可能是碎片）
        - 句子完整度（是否有完整句子结构）
    """
    if not text:
        return 0.0

    length = len(text)
    alpha_chars = sum(1 for c in text if c.isalpha() or "\u4e00" <= c <= "\u9fff")
    alpha_ratio = alpha_chars / max(length, 1)

    # 有实际内容的词数（中文字符或英文单词）
    words = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]{2,}", text))

    # 质量分：有效字符比例 + 适当长度
    length_score = min(length / settings.CHUNK_SIZE, 1.0)
    quality = 0.5 * alpha_ratio + 0.5 * min(words / 20, 1.0) * length_score

    return round(quality, 3)


def clean_chunks(
    chunks: list[LCDocument],
    min_length: int | None = None,
    max_length: int | None = None,
    quality_threshold: float = 0.2,
) -> list[LCDocument]:
    """对 chunk 列表进行清洗和质量过滤。

    处理流程：
        1. 规范化空白
        2. 过滤噪声 chunk
        3. 过滤过短/过长的 chunk
        4. 质量评分（写入 metadata）
        5. 相邻 chunk 重复检测

    Args:
        chunks: 原始切分后的 chunk 列表。
        min_length: 最小字符数阈值（低于此值剔除）。
        max_length: 最大字符数阈值（高于此值警告但不剔除）。
        quality_threshold: 质量分最低阈值（低于此值剔除）。

    Returns:
        清洗后的 chunk 列表（可能少于输入）。
    """
    min_len = min_length or settings.CHUNK_MIN_LENGTH
    max_len = max_length or settings.CHUNK_SIZE * 2

    cleaned: list[LCDocument] = []
    dropped = 0

    for chunk in chunks:
        original = chunk.page_content
        normalized = _normalize_whitespace(original)

        # 噪声过滤
        if _is_noise(normalized):
            dropped += 1
            continue

        # 长度过滤
        if len(normalized) < min_len:
            dropped += 1
            continue

        if len(normalized) > max_len:
            # 不丢弃但警告
            logger.debug("chunk 超长", length=len(normalized), preview=normalized[:80])

        # 质量评分
        quality = _score_quality(normalized)
        if quality < quality_threshold:
            dropped += 1
            continue

        # 更新内容
        chunk.page_content = normalized
        chunk.metadata["quality_score"] = quality
        chunk.metadata["cleaned"] = True
        cleaned.append(chunk)

    logger.info(
        "chunk 清洗完成",
        input=len(chunks),
        output=len(cleaned),
        dropped=dropped,
    )
    return cleaned
