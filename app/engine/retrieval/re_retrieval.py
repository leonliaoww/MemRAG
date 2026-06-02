"""低置信度二次检索机制。

当首轮检索的 top 结果置信度不足时，自动触发策略升级的二次检索：

策略链（逐级升级）：
    第1轮：用户原始查询 → 混合检索 → 重排序 → 检查分数
    第2轮：LLM 改写查询 → 混合检索 → 重排序 → 合并去重 → 再次检查
    第3轮（最后）：纯关键词回退 → 不再重排序 → 直接合并

每轮检索后评估 top-1 的置信度分数：
    - cross_encoder_score ≥ 阈值 → 停止，进入生成
    - cross_encoder_score < 阈值  → 升级策略，继续下一轮

设计原则：
    - 渐进式升级：不给用户增加延迟除非必要
    - 结果合并去重：每轮结果按 RRF 融合，避免冗余
    - 最终降级标记：多轮后仍低分，标记为 low_confidence，由 LLM 判断是否回答
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.generation.llm import get_singleton_llm
from app.engine.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from app.engine.retrieval.reranker import rerank
from app.engine.retrieval.vector_store import search_similar
from app.schemas.query import RetrievalMode

logger = get_logger(__name__)


@dataclass
class ReRetrievalResult:
    """二次检索的完整结果。"""
    documents: list[LCDocument] = field(default_factory=list)
    """最终用于生成的文档列表（多轮去重合并后）。"""
    rounds_used: int = 1
    """实际执行的检索轮数。"""
    low_confidence: bool = False
    """是否仍处于低置信度状态（多轮后仍未达标）。"""
    strategy_path: list[str] = field(default_factory=list)
    """每轮使用的策略名称（用于监控和调试）。"""
    queries_used: list[str] = field(default_factory=list)
    """每轮使用的查询字符串。"""


# ── 置信度评估 ────────────────────────────────────────

def _get_top_score(documents: list[LCDocument]) -> float:
    """获取文档列表中的最高置信度分数。

    优先级：cross_encoder_score > relevance_score > 0.0
    """
    if not documents:
        return 0.0

    best = 0.0
    for doc in documents:
        score = doc.metadata.get("cross_encoder_score")
        if score is None:
            score = doc.metadata.get("relevance_score", 0.0)
        if score and score > best:
            best = float(score)
    return best


def _is_low_confidence(documents: list[LCDocument]) -> bool:
    """判断检索结果是否低置信度。"""
    return _get_top_score(documents) < settings.LOW_CONFIDENCE_THRESHOLD


# ── 查询改写策略 ──────────────────────────────────────

QUERY_REWRITE_FOR_RETRIEVAL = """你是一个查询优化专家。用户的原始问题检索到的文档相关度不足，
请将以下问题改写为更适合文档检索的形式。

改写规则：
1. 提取问题的核心实体和关键概念
2. 如果问题模糊，补充可能的近义词或相关术语
3. 如果是复合问题，拆解为最核心的子问题
4. 只输出改写后的问题，不要加任何解释

原始问题：{question}

改写后的问题："""


async def _rewrite_query(question: str) -> str:
    """使用 LLM 将低置信度问题改写为更精确的检索查询。

    与 QUERY_REWRITE_PROMPT 不同，这个 prompt 侧重于
    "检索失败后的补救"，会尝试提取核心实体和扩展关键术语。
    """
    try:
        llm = get_singleton_llm()
        from langchain_core.messages import HumanMessage

        prompt = QUERY_REWRITE_FOR_RETRIEVAL.format(question=question)
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = str(result.content).strip()
        logger.info("二次检索查询改写", original=question[:80], rewritten=rewritten[:80])
        return rewritten
    except Exception as exc:
        logger.warning("查询改写失败，使用原始查询", error=str(exc))
        return question


# ── 结果合并去重 ──────────────────────────────────────

def _merge_and_dedup(
    existing: list[LCDocument],
    new: list[LCDocument],
    top_k: int,
) -> list[LCDocument]:
    """合并两轮检索结果并去重。

    使用 RRF 融合确保两轮结果公平排序。
    去重依据：文档的 source + chunk_index。
    """
    if not new:
        return existing

    if not existing:
        return new[:top_k]

    # 以 (doc, score) 格式传入 RRF
    existing_pairs = [
        (doc, doc.metadata.get("cross_encoder_score") or doc.metadata.get("relevance_score", 0.0))
        for doc in existing
    ]
    new_pairs = [
        (doc, doc.metadata.get("cross_encoder_score") or doc.metadata.get("relevance_score", 0.0))
        for doc in new
    ]

    merged = reciprocal_rank_fusion(existing_pairs, new_pairs)

    # 辅助去重：同 source 同 chunk_index 的只保留得分更高的
    seen: dict[str, LCDocument] = {}
    for doc in merged:
        key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('chunk_index', '')}"
        if key not in seen:
            seen[key] = doc
        else:
            existing_score = seen[key].metadata.get("rrf_score", 0)
            new_score = doc.metadata.get("rrf_score", 0)
            if new_score > existing_score:
                seen[key] = doc

    deduped = sorted(
        seen.values(),
        key=lambda d: d.metadata.get("rrf_score", 0),
        reverse=True,
    )

    logger.debug(
        "结果合并去重",
        existing=len(existing),
        new=len(new),
        merged=len(merged),
        deduped=len(deduped),
    )
    return deduped[:top_k]


# ── 主入口 ────────────────────────────────────────────

async def retrieve_with_fallback(
    question: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
) -> ReRetrievalResult:
    """带低置信度二次检索的智能检索入口。

    检索流程：
        第1轮：原始查询 + 混合检索 + 重排序
        第2轮（如需要）：LLM 改写查询 + 混合检索 + 重排序 + 合并
        第3轮（如需要）：纯关键词回退 + 直接合并（不重排序以节省时间）

    Args:
        question: 用户问题。
        collection_name: Chroma collection。
        top_k: 最终返回数量。
        retrieval_mode: 检索模式。

    Returns:
        ReRetrievalResult 包含合并后的文档列表和检索元数据。
    """
    if not settings.RE_RETRIEVAL_ENABLED:
        # 二次检索关闭 → 单轮直接返回
        result = await _single_retrieval_round(
            question, collection_name, top_k, retrieval_mode, "原始查询"
        )
        return ReRetrievalResult(
            documents=result,
            rounds_used=1,
            strategy_path=["原始查询"],
            queries_used=[question],
        )

    k = top_k or settings.RETRIEVAL_TOP_K
    max_rounds = settings.RE_RETRIEVAL_MAX_ROUNDS

    all_documents: list[LCDocument] = []
    strategy_path: list[str] = []
    queries_used: list[str] = []
    current_query = question

    for round_idx in range(max_rounds):
        # ── 选择策略 ──
        if round_idx == 0:
            strategy = "原始查询 + 混合检索"
            mode = retrieval_mode
        elif round_idx == 1:
            strategy = "LLM 改写查询"
            mode = RetrievalMode.HYBRID
            current_query = await _rewrite_query(question)
        else:
            strategy = "纯关键词回退"
            mode = RetrievalMode.KEYWORD

        strategy_path.append(strategy)
        queries_used.append(current_query)

        logger.info(
            "检索轮次",
            round=round_idx + 1,
            strategy=strategy,
            query=current_query[:80],
        )

        # ── 执行检索 ──
        round_docs = await _single_retrieval_round(
            current_query, collection_name, k * 2, mode, strategy
        )

        # 合并去重
        all_documents = _merge_and_dedup(all_documents, round_docs, k)

        if not all_documents:
            continue  # 无结果，继续下一轮

        # ── 检查置信度 ──
        if not _is_low_confidence(all_documents):
            logger.info(
                "置信度达标，停止检索",
                round=round_idx + 1,
                top_score=_get_top_score(all_documents),
            )
            return ReRetrievalResult(
                documents=all_documents[:k],
                rounds_used=round_idx + 1,
                strategy_path=strategy_path,
                queries_used=queries_used,
            )

    # 多轮后仍低置信度 → 降级标记
    logger.warning(
        "多轮检索后仍低置信度",
        rounds=max_rounds,
        top_score=_get_top_score(all_documents),
    )
    return ReRetrievalResult(
        documents=all_documents[:k],
        rounds_used=max_rounds,
        low_confidence=True,
        strategy_path=strategy_path,
        queries_used=queries_used,
    )


async def _single_retrieval_round(
    query: str,
    collection_name: str | None,
    top_k: int,
    mode: RetrievalMode,
    strategy: str,
) -> list[LCDocument]:
    """执行单轮检索（检索 + 可选重排序）。

    根据策略不同，选择不同的检索后端。
    """
    # 检索
    if mode == RetrievalMode.HYBRID and settings.HYBRID_SEARCH_ENABLED:
        docs = await hybrid_search(query, top_k=top_k, collection_name=collection_name)
    elif mode == RetrievalMode.KEYWORD:
        # 纯关键词：走 BM25，不回退向量
        from app.engine.retrieval.bm25_index import get_bm25_index
        bm25 = get_bm25_index(collection_name)
        if bm25.doc_count > 0:
            bm25_raw = bm25.search(query, top_k=top_k)
            docs = [doc for doc, _ in bm25_raw]
        else:
            docs = []
    else:
        docs = await search_similar(query, top_k=top_k, collection_name=collection_name)

    if not docs:
        return []

    # 重排序（关键词模式跳过，因为 BM25 分数已经够用）
    if settings.RERANK_ENABLED and mode != RetrievalMode.KEYWORD:
        docs = rerank(query, docs, top_k=top_k)

    logger.debug(
        "单轮检索完成",
        strategy=strategy,
        docs=len(docs),
        top_score=_get_top_score(docs),
    )
    return docs
