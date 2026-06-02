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

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.retrieval.vector_store import search_similar

logger = get_logger(__name__)


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
) -> list[LCDocument]:
    """执行混合检索：向量 + BM25 → RRF 融合。

    BM25 使用持久化索引（bm25_index.py），应用启动时自动加载，
    文档摄取后自动更新，无需每次检索时重建。

    当 BM25 索引为空时（首次使用、无文档），BM25 路自动跳过，
    降级为纯向量检索。

    Args:
        query: 用户查询。
        top_k: 最终返回的文档数。
        collection_name: Chroma collection。

    Returns:
        RRF 融合后的排序文档列表。
    """
    from app.engine.retrieval.bm25_index import get_bm25_index

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

    # ── BM25 检索（持久化索引）──
    bm25_results: list[tuple[LCDocument, float]] = []
    try:
        bm25_index = get_bm25_index(collection_name)
        if bm25_index.doc_count > 0:
            bm25_raw = bm25_index.search(query, top_k=fetch_k)
            bm25_results = [(doc, score) for doc, score in bm25_raw]
        else:
            logger.debug("BM25 索引为空，跳过", collection=collection_name)
    except Exception as exc:
        logger.warning("BM25 检索失败，降级为纯向量检索", error=str(exc))

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
