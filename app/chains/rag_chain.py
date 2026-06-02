"""标准 RAG 链 — 核心的检索增强生成管道。

管道流程：
    1. 检索相关文档（向量 / 混合检索）
    2. CrossEncoder 重排序候选文档
    3. 格式化为 LLM 可读的上下文字符串
    4. 构建 prompt（上下文 + 问题）
    5. LLM 生成答案
    6. 从答案中提取并匹配引用来源

每个阶段都独立可追踪，便于调试和后续迁移到 LangGraph。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.generation.citations import build_citations, format_context
from app.engine.generation.llm import generate, generate_stream
from app.engine.generation.llm import get_singleton_llm
from app.engine.generation.prompts import QUERY_REWRITE_PROMPT, RAG_PROMPT
from app.engine.retrieval.hybrid import hybrid_search
from app.engine.retrieval.reranker import rerank
from app.engine.retrieval.vector_store import search_similar
from app.schemas.query import RetrievalMode, SourceCitation

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """RAG 查询的完整结果 — 包含答案、来源、检索模式等。"""
    question: str
    answer: str = ""
    retrieved_docs: list[LCDocument] = field(default_factory=list)
    citations: list[SourceCitation] = field(default_factory=list)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    tokens_used: int | None = None


async def run_rag(
    question: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
) -> RAGResult:
    """执行完整的 RAG 管道。

    这是系统最核心的函数 — 从用户问题到带引用的答案。

    Args:
        question: 用户问题。
        collection_name: 目标 Chroma collection（对应知识库）。
        top_k: 检索文档数量。
        retrieval_mode: 检索策略（向量/混合/关键词）。

    Returns:
        RAGResult 包含答案、引用来源和管道元数据。
    """
    k = top_k or settings.RETRIEVAL_TOP_K

    # ── 第0步：查询改写 ──
    # 将用户问题改写为更适合检索的形式
    search_query = question
    if settings.QUERY_REWRITE_ENABLED:
        from langchain_core.output_parsers import StrOutputParser
        llm = get_singleton_llm()
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        search_query = await rewrite_chain.ainvoke({"chat_history": "", "question": question})
        logger.debug("查询改写", original=question, rewritten=search_query)

    # ── 第1步：检索（带低置信度二次检索）──
    if settings.RE_RETRIEVAL_ENABLED:
        from app.engine.retrieval.re_retrieval import retrieve_with_fallback

        re_result = await retrieve_with_fallback(
            question=search_query,
            collection_name=collection_name,
            top_k=k,
            retrieval_mode=retrieval_mode,
        )
        retrieved = re_result.documents
        if re_result.low_confidence:
            logger.warning(
                "RAG 低置信度降级",
                rounds=re_result.rounds_used,
                strategies=re_result.strategy_path,
            )
        logger.debug(
            "RAG检索（二次检索）",
            docs=len(retrieved),
            rounds=re_result.rounds_used,
            strategies=re_result.strategy_path,
        )
    else:
        if retrieval_mode == RetrievalMode.HYBRID and settings.HYBRID_SEARCH_ENABLED:
            retrieved = await hybrid_search(search_query, top_k=k, collection_name=collection_name)
            logger.debug("RAG检索", mode="hybrid", docs=len(retrieved))
        else:
            retrieved = await search_similar(search_query, top_k=k, collection_name=collection_name)
            logger.debug("RAG检索", mode="vector", docs=len(retrieved))

        # 重排序
        if settings.RERANK_ENABLED and retrieved:
            retrieved = rerank(search_query, retrieved, top_k=k)

    # 没有检索到任何文档 → 直接返回
    if not retrieved:
        return RAGResult(
            question=question,
            answer="未找到与此问题相关的文档，无法回答。",
            retrieval_mode=retrieval_mode,
        )

    # ── 第3步：构建上下文 ──
    context = format_context(retrieved)

    # ── 第4步和第5步：构建 Prompt + LLM 生成 ──
    # 直接操作消息对象而非使用 chain.invoke()，便于逐阶段调试和日志记录
    system_msg = RAG_PROMPT.messages[0].prompt.template  # type: ignore[union-attr]
    user_msg = RAG_PROMPT.messages[1].prompt.template.format(  # type: ignore[union-attr]
        context=context, question=question
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]

    answer = await generate(messages)

    # ── 第6步：引用追踪 ──
    citations = build_citations(answer, retrieved)

    return RAGResult(
        question=question,
        answer=answer,
        retrieved_docs=retrieved,
        citations=citations,
        retrieval_mode=retrieval_mode,
    )


async def run_rag_stream(
    question: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
):
    """执行 RAG 管道并流式输出 — 供 SSE 端点使用。

    与 run_rag() 逻辑一致，只是生成阶段使用流式接口。

    Yields:
        格式为 {"token": str, "done": bool, "citations": list | None} 的字典。
    """
    k = top_k or settings.RETRIEVAL_TOP_K

    # 查询改写
    search_query = question
    if settings.QUERY_REWRITE_ENABLED:
        from langchain_core.output_parsers import StrOutputParser
        llm = get_singleton_llm()
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        search_query = await rewrite_chain.ainvoke({"chat_history": "", "question": question})
        logger.debug("查询改写", original=question, rewritten=search_query)

    # 检索（带低置信度二次检索）
    if settings.RE_RETRIEVAL_ENABLED:
        from app.engine.retrieval.re_retrieval import retrieve_with_fallback

        re_result = await retrieve_with_fallback(
            question=search_query,
            collection_name=collection_name,
            top_k=k,
            retrieval_mode=retrieval_mode,
        )
        retrieved = re_result.documents
    else:
        if retrieval_mode == RetrievalMode.HYBRID and settings.HYBRID_SEARCH_ENABLED:
            retrieved = await hybrid_search(search_query, top_k=k, collection_name=collection_name)
        else:
            retrieved = await search_similar(search_query, top_k=k, collection_name=collection_name)

        if settings.RERANK_ENABLED and retrieved:
            retrieved = rerank(question, retrieved, top_k=k)

    if not retrieved:
        yield {"token": "未找到相关文档。", "done": True, "citations": []}
        return

    # 构建上下文和消息
    context = format_context(retrieved)

    system_msg = RAG_PROMPT.messages[0].prompt.template  # type: ignore[union-attr]
    user_msg = RAG_PROMPT.messages[1].prompt.template.format(  # type: ignore[union-attr]
        context=context, question=question
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]

    # 流式生成
    full_answer: list[str] = []
    async for token in generate_stream(messages):
        full_answer.append(str(token))
        yield {"token": str(token), "done": False}

    # 最后一个 chunk 附带引用信息
    answer_text = "".join(full_answer)
    citations = build_citations(answer_text, retrieved)
    yield {"token": "", "done": True, "citations": citations}
