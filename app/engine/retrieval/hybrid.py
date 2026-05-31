"""混合检索 — 向量语义搜索 + BM25 关键词检索。

检索策略：
    1. 向量搜索（OpenAI embeddings 余弦相似度）→ 取 top_k * 2 候选
    2. BM25 关键词搜索（稀疏检索）→ 取 top_k * 2 候选
    3. 通过 Reciprocal Rank Fusion（RRF）算法融合两路结果
    4. 返回 top_k 个最终结果

RRF 优势：
    - 无需调参即可融合不同量纲的分数
    - 对排名位置敏感，对绝对分数不敏感
    - 业界验证的有效融合策略（k=60 为文献推荐值）

注意：BM25 需要加载全量文档构建索引，适合中小规模（<100k 文档）。
大规模场景建议使用 Elasticsearch 或外部 BM25 服务。
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document as LCDocument
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.retrieval.vector_store import search_similar

logger = get_logger(__name__)


def _build_bm25_index(documents: list[LCDocument]) -> BM25Okapi:
    """从文档列表构建内存 BM25 索引。

    使用简单的空格分词（对小写文本）。
    中文场景建议后续接入 jieba 分词以提升召回率。

    Args:
        documents: 待索引的文档列表。

    Returns:
        BM25Okapi 索引对象。
    """
    tokenized = [doc.page_content.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def reciprocal_rank_fusion(
    vector_results: list[tuple[LCDocument, float]],
    bm25_results: list[tuple[LCDocument, float]],
    k: int = 60,
) -> list[LCDocument]:
    """用 RRF 算法融合两路检索结果。

    RRF 公式：score(d) = Σ 1/(k + rank_i(d))
    其中 rank_i(d) 是文档 d 在检索引擎 i 中的排名位置。

    Args:
        vector_results: 向量检索结果，每项为 (doc, score) 元组。
        bm25_results: BM25 检索结果。
        k: RRF 常数（默认 60，平滑排名差异）。

    Returns:
        按 RRF 分数降序排列的文档列表，每个文档的 metadata.rrf_score 记录融合分数。
    """
    doc_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, LCDocument] = {}

    # 第一遍：累加向量检索的 RRF 分数
    for rank, (doc, _) in enumerate(vector_results, start=1):
        doc_id = doc.metadata.get("chunk_index", id(doc))
        key = f"{doc.metadata.get('source', '')}_{doc_id}"
        doc_scores[key] += 1.0 / (k + rank)
        doc_map[key] = doc

    # 第二遍：累加 BM25 检索的 RRF 分数
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        doc_id = doc.metadata.get("chunk_index", id(doc))
        key = f"{doc.metadata.get('source', '')}_{doc_id}"
        doc_scores[key] += 1.0 / (k + rank)
        doc_map[key] = doc

    # 按 RRF 总分降序排列
    sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)  # type: ignore[arg-type]

    merged: list[LCDocument] = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = doc_scores[key]
        merged.append(doc)

    return merged


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    all_documents: list[LCDocument] | None = None,
) -> list[LCDocument]:
    """执行混合检索：向量 + BM25 → RRF 融合。

    当 all_documents 为 None 时，BM25 路被跳过，
    自动降级为纯向量检索（不报错）。

    Args:
        query: 用户查询。
        top_k: 最终返回的文档数。
        collection_name: Chroma collection。
        all_documents: 用于构建 BM25 索引的全量文档列表。
                       为 None 时仅使用向量检索。

    Returns:
        RRF 融合后的排序文档列表。
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    fetch_k = k * 2  # 每路多取一些，给融合留余量

    # ── 向量检索 ──
    vector_results_raw = await search_similar(
        query, top_k=fetch_k, collection_name=collection_name
    )
    vector_results: list[tuple[LCDocument, float]] = [
        (doc, doc.metadata.get("relevance_score", 0.0))
        for doc in vector_results_raw
    ]

    # ── BM25 检索 ──
    bm25_results: list[tuple[LCDocument, float]] = []
    if all_documents:
        bm25 = _build_bm25_index(all_documents)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # 取 BM25 分数最高的 fetch_k 个
        scored = sorted(
            zip(all_documents, bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:fetch_k]
        bm25_results = scored
    else:
        logger.debug("跳过 BM25", reason="未提供全量文档列表")

    # ── RRF 融合 ──
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    top_docs = fused[:k]

    logger.info(
        "混合检索完成",
        query=query[:80],
        vector_hits=len(vector_results),
        bm25_hits=len(bm25_results),
        fused_results=len(top_docs),
    )
    return top_docs
