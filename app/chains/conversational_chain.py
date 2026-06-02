"""多轮对话 RAG 链 — 支持带上下文的检索增强对话。

功能：
    - 对话历史管理（滑动窗口，保留最近 N 轮）
    - 上下文感知检索（将上一轮回答作为附加上下文）
    - 历史和上下文联合生成

与标准 RAG 链的区别：
    - Prompt 中注入了聊天记录
    - 检索时考虑上一轮对话内容作为辅助上下文
    - 返回结果包含完整的历史记录

LangGraph 迁移路径：
    每个函数对应一个 LangGraph 节点，历史管理可转为状态图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engine.generation.citations import build_citations, format_context
from app.engine.generation.llm import generate, generate_stream, get_singleton_llm
from app.engine.generation.prompts import CONVERSATIONAL_PROMPT, QUERY_REWRITE_PROMPT
from app.engine.retrieval.hybrid import hybrid_search
from app.engine.retrieval.reranker import rerank
from app.engine.retrieval.vector_store import search_similar as vector_search
from app.schemas.query import RetrievalMode, SourceCitation

logger = get_logger(__name__)

# 对话历史中保留的最大轮数（一轮 = 用户问题 + 助手回答）
MAX_HISTORY_TURNS = 5


@dataclass
class ConversationTurn:
    """单轮对话记录。"""
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ConversationalRAGResult:
    """多轮对话 RAG 查询的完整结果。"""
    question: str
    answer: str = ""
    retrieved_docs: list[LCDocument] = field(default_factory=list)
    citations: list[SourceCitation] = field(default_factory=list)
    history: list[ConversationTurn] = field(default_factory=list)
    tokens_used: int | None = None


def format_history(history: list[ConversationTurn]) -> str:
    """将对话历史格式化为 LLM 可读的字符串。

    只保留最近的 MAX_HISTORY_TURNS 轮对话，
    防止 prompt 过长导致 token 溢出。

    Args:
        history: 完整对话历史记录。

    Returns:
        格式化的历史文本，每行格式为 "角色: 内容"。
    """
    if not history:
        return "（无历史对话）"

    lines: list[str] = []
    # 只取最近 N 轮（每轮 2 条：用户 + 助手）
    for turn in history[-MAX_HISTORY_TURNS * 2:]:
        label = "用户" if turn.role == "user" else "助手"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


async def run_conversational_rag(
    question: str,
    history: list[ConversationTurn],
    collection_name: str | None = None,
    top_k: int | None = None,
) -> ConversationalRAGResult:
    """执行多轮对话 RAG 管道。

    检索时会利用最近的对话内容来增强查询，
    以处理代词指代和省略等对话特有现象。

    Args:
        question: 用户的最新问题。
        history: 之前的对话记录。
        collection_name: Chroma collection。
        top_k: 检索数量。

    Returns:
        ConversationalRAGResult 带答案、引用和历史。
    """
    k = top_k or settings.RETRIEVAL_TOP_K

    # ── 第1步：上下文感知检索 + 查询改写 ──
    # 将上一轮助手的回答作为辅助上下文拼接到查询中
    history_context = ""
    if history:
        last_turn = history[-1]
        if last_turn.role == "assistant":
            history_context = last_turn.content[:200]

    # 使用 LLM 将问题改写为适合检索的形式
    search_query = question
    if settings.QUERY_REWRITE_ENABLED:
        from langchain_core.output_parsers import StrOutputParser
        llm = get_singleton_llm()
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        search_query = await rewrite_chain.ainvoke(
            {"chat_history": history_context, "question": question}
        )
        logger.debug("查询改写", original=question, rewritten=search_query)
    elif history_context:
        # 如果未启用查询改写，回退到简单拼接
        search_query = f"{question}（上下文：{history_context}）"

    # 检索（带低置信度二次检索）
    if settings.RE_RETRIEVAL_ENABLED:
        from app.engine.retrieval.re_retrieval import retrieve_with_fallback
        from app.schemas.query import RetrievalMode

        re_result = await retrieve_with_fallback(
            question=search_query,
            collection_name=collection_name,
            top_k=k,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        retrieved = re_result.documents
    else:
        if settings.HYBRID_SEARCH_ENABLED:
            retrieved = await hybrid_search(search_query, top_k=k, collection_name=collection_name)
        else:
            retrieved = await vector_search(search_query, top_k=k, collection_name=collection_name)

        if settings.RERANK_ENABLED and retrieved:
            retrieved = rerank(question, retrieved, top_k=k)

    if not retrieved:
        return ConversationalRAGResult(
            question=question,
            answer="我没有足够的上下文来回答这个问题。请换一种方式提问或提供更多信息。",
            history=history,
        )

    # ── 第3步和第4步：构建上下文 + Prompt ──
    context = format_context(retrieved)
    chat_history_str = format_history(history)

    system_msg = CONVERSATIONAL_PROMPT.messages[0].prompt.template  # type: ignore[union-attr]
    user_msg = CONVERSATIONAL_PROMPT.messages[1].prompt.template.format(  # type: ignore[union-attr]
        chat_history=chat_history_str,
        context=context,
        question=question,
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]

    # ── 第5步：生成 ──
    answer = await generate(messages)

    # ── 第6步：引用 ──
    citations = build_citations(answer, retrieved)

    return ConversationalRAGResult(
        question=question,
        answer=answer,
        retrieved_docs=retrieved,
        citations=citations,
        history=history,
    )


async def run_conversational_rag_stream(
    question: str,
    history: list[ConversationTurn],
    collection_name: str | None = None,
    top_k: int | None = None,
):
    """流式版本的多轮对话 RAG — 供 SSE 端点使用。

    与 run_conversational_rag() 逻辑一致，生成阶段使用流式接口。
    """
    k = top_k or settings.RETRIEVAL_TOP_K

    # 查询改写
    history_context = ""
    if history:
        last_turn = history[-1]
        if last_turn.role == "assistant":
            history_context = last_turn.content[:200]

    search_query = question
    if settings.QUERY_REWRITE_ENABLED:
        from langchain_core.output_parsers import StrOutputParser
        llm = get_singleton_llm()
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        search_query = await rewrite_chain.ainvoke(
            {"chat_history": history_context, "question": question}
        )
        logger.debug("查询改写", original=question, rewritten=search_query)
    elif history_context:
        search_query = f"{question}（上下文：{history_context}）"

    # 检索（带低置信度二次检索）
    if settings.RE_RETRIEVAL_ENABLED:
        from app.engine.retrieval.re_retrieval import retrieve_with_fallback
        from app.schemas.query import RetrievalMode

        re_result = await retrieve_with_fallback(
            question=search_query,
            collection_name=collection_name,
            top_k=k,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        retrieved = re_result.documents
    else:
        if settings.HYBRID_SEARCH_ENABLED:
            retrieved = await hybrid_search(search_query, top_k=k, collection_name=collection_name)
        else:
            retrieved = await vector_search(search_query, top_k=k, collection_name=collection_name)

        if settings.RERANK_ENABLED and retrieved:
            retrieved = rerank(question, retrieved, top_k=k)

    if not retrieved:
        yield {"token": "我没有足够的上下文来回答这个问题。", "done": True, "citations": []}
        return

    context = format_context(retrieved)
    chat_history_str = format_history(history)

    system_msg = CONVERSATIONAL_PROMPT.messages[0].prompt.template  # type: ignore[union-attr]
    user_msg = CONVERSATIONAL_PROMPT.messages[1].prompt.template.format(  # type: ignore[union-attr]
        chat_history=chat_history_str,
        context=context,
        question=question,
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]

    full_answer: list[str] = []
    async for token in generate_stream(messages):
        full_answer.append(str(token))
        yield {"token": str(token), "done": False}

    answer_text = "".join(full_answer)
    citations = build_citations(answer_text, retrieved)
    yield {"token": "", "done": True, "citations": citations}
