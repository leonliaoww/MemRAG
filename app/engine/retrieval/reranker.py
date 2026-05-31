"""CrossEncoder 重排序 — 提升检索精度。

原理：
    向量检索（双塔模型）速度快但精度有限，因为 query 和 document
    分别编码后仅通过点积/余弦交互。
    CrossEncoder 将 (query, document) 对联合输入，充分交叉注意力，
    因此精度显著更高，但速度较慢。

策略：
    粗筛（向量/BM25，毫秒级）→ 精排（CrossEncoder，百毫秒级）
    先用廉价检索召回候选集，再用 CrossEncoder 精排 top 结果。

模型加载方式：
    1. 设置 RERANK_MODEL_PATH → 从本地目录离线加载（不需要网络）
    2. 不设置 → 从 HuggingFace Hub 自动下载并缓存

模型文件获取（首次需要联网下载到本地）：
    git lfs install
    git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 ./models/reranker/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# CrossEncoder 模型懒加载
_cross_encoder = None


def _load_from_local(path: str):
    """从本地目录加载 CrossEncoder 模型（离线模式）。

    模型目录应包含 config.json、pytorch_model.bin（或 model.safetensors）、
    tokenizer_config.json 等文件。
    """
    from sentence_transformers import CrossEncoder

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"重排序模型目录不存在: {path}\n"
            f"请先从 HuggingFace 下载模型：\n"
            f"  git lfs install\n"
            f"  git clone https://huggingface.co/{settings.RERANK_MODEL} {path}"
        )
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"模型目录缺少 config.json: {path}\n请确认目录是完整的 HuggingFace 模型仓库"
        )

    logger.info("从本地加载重排序模型", path=path)
    return CrossEncoder(str(model_path))


def _load_from_hub(model_name: str):
    """从 HuggingFace Hub 下载并加载模型（需要网络）。"""
    from sentence_transformers import CrossEncoder

    logger.info("从 HuggingFace 加载重排序模型", model=model_name)
    return CrossEncoder(model_name)


def _get_cross_encoder():
    """懒加载 CrossEncoder 模型。

    加载优先级：
        1. RERANK_MODEL_PATH 有值 → 从本地路径加载（离线）
        2. 否则 → 从 HuggingFace Hub 下载

    模型约 80MB，加载后常驻内存。
    """
    global _cross_encoder
    if _cross_encoder is None and settings.RERANK_ENABLED:
        try:
            # 优先从本地路径加载
            if settings.RERANK_MODEL_PATH:
                _cross_encoder = _load_from_local(settings.RERANK_MODEL_PATH)
            else:
                _cross_encoder = _load_from_hub(settings.RERANK_MODEL)
        except ImportError:
            logger.warning("重排序不可用", reason="sentence_transformers 未安装")
        except FileNotFoundError as exc:
            logger.error("重排序模型加载失败", error=str(exc))
            # 不回退到 HuggingFace——用户明确配置了本地路径就应使用本地
        except Exception as exc:
            logger.error("重排序模型加载失败", error=str(exc))
    return _cross_encoder


def rerank(
    query: str,
    documents: list[LCDocument],
    top_k: int | None = None,
) -> list[LCDocument]:
    """使用 CrossEncoder 对候选文档重新排序。

    每个文档的 metadata.cross_encoder_score 会被更新为重排序分数。

    Args:
        query: 用户查询字符串。
        documents: 初筛后的候选文档列表。
        top_k: 重排序后保留的文档数量（默认全部保留）。

    Returns:
        按 CrossEncoder 相关度降序排列的文档列表。
        如果重排序不可用或禁用，返回原始顺序。
    """
    # 配置关闭重排序 或 空列表 → 直接返回
    if not settings.RERANK_ENABLED or not documents:
        return documents

    model = _get_cross_encoder()
    if model is None:
        return documents

    k = top_k or len(documents)
    k = min(k, len(documents))

    # 构建 (query, document) 配对列表
    pairs = [(query, doc.page_content) for doc in documents]

    try:
        scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]

        # 将 CrossEncoder 分数写入每个文档的元数据
        for doc, score in zip(documents, scores):
            doc.metadata["cross_encoder_score"] = float(score)

        # 按分数降序排列
        reranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_docs = [doc for doc, _ in reranked[:k]]

        logger.debug("重排序完成", input=len(documents), output=len(top_docs))
        return top_docs

    except Exception as exc:
        logger.error("重排序失败，返回原始顺序", error=str(exc))
        return documents[:k]
