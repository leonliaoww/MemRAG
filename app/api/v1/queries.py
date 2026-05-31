"""查询 API — 核心 RAG 问答端点。

端点：
    POST /api/v1/queries/ask         — 标准 RAG 问答（非流式）
    POST /api/v1/queries/ask/stream  — 标准 RAG 问答（SSE 流式）
    POST /api/v1/queries/chat        — 多轮对话 RAG（非流式）
    POST /api/v1/queries/chat/stream — 多轮对话 RAG（SSE 流式）

对话会话管理：
    MVP 阶段使用内存字典存储会话历史。
    生产环境应替换为 Redis hash，并设置 TTL 过期。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.chains.conversational_chain import (
    ConversationTurn,
    run_conversational_rag,
    run_conversational_rag_stream,
)
from app.chains.rag_chain import run_rag, run_rag_stream
from app.core.logging_config import get_logger
from app.schemas.query import AskRequest, AskResponse, RetrievalMode, StreamChunk

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/queries", tags=["queries"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """对知识库提出一个问题，获取带来源引用的完整答案。

    请求示例：
        {"question": "季度营收是多少？", "top_k": 5, "retrieval_mode": "hybrid"}

    返回带 citations 的完整答案。
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    result = await run_rag(
        question=request.question,
        collection_name=request.knowledge_base_id,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
    )

    return AskResponse(
        answer=result.answer,
        sources=result.citations,
        retrieval_mode=result.retrieval_mode,
        tokens_used=result.tokens_used,
    )


@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """流式 RAG 问答 — 使用 Server-Sent Events 逐 token 返回。

    每个事件包含一个 JSON 格式的 StreamChunk。
    最后一个事件的 done=True，并附带完整的 sources 列表。
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def event_generator():
        async for chunk in run_rag_stream(
            question=request.question,
            collection_name=request.knowledge_base_id,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
        ):
            yield {
                "event": "token",
                "data": StreamChunk(
                    content=chunk["token"] if not chunk["done"] else "",
                    done=chunk["done"],
                    sources=chunk.get("citations"),
                ).model_dump_json(),
            }

    return EventSourceResponse(event_generator())


# ── 内存会话存储（生产环境应使用 Redis）───────────────
_conversations: dict[str, list[ConversationTurn]] = {}


@router.post("/chat", response_model=AskResponse)
async def chat(request: AskRequest) -> AskResponse:
    """多轮对话 RAG — 支持连续追问和上下文保持。

    使用 knowledge_base_id 作为会话标识。
    生产环境应使用独立的 session_id 参数。
    """
    session_id = request.knowledge_base_id or "default"
    history = _conversations.get(session_id, [])

    result = await run_conversational_rag(
        question=request.question,
        history=history,
        collection_name=request.knowledge_base_id,
        top_k=request.top_k,
    )

    # 更新对话历史
    history.append(ConversationTurn(role="user", content=request.question))
    history.append(ConversationTurn(role="assistant", content=result.answer))
    _conversations[session_id] = history

    return AskResponse(
        answer=result.answer,
        sources=result.citations,
        retrieval_mode=request.retrieval_mode,
    )


@router.post("/chat/stream")
async def chat_stream(request: AskRequest):
    """流式多轮对话 RAG — SSE 逐 token 返回。"""
    session_id = request.knowledge_base_id or "default"
    history = _conversations.get(session_id, [])

    async def event_generator():
        full_answer: list[str] = []
        async for chunk in run_conversational_rag_stream(
            question=request.question,
            history=history,
            collection_name=request.knowledge_base_id,
            top_k=request.top_k,
        ):
            yield {
                "event": "token",
                "data": StreamChunk(
                    content=chunk["token"] if not chunk["done"] else "",
                    done=chunk["done"],
                    sources=chunk.get("citations"),
                ).model_dump_json(),
            }
            if not chunk["done"]:
                full_answer.append(chunk["token"])

        # 流式完成后更新对话历史
        answer_text = "".join(full_answer)
        history.append(ConversationTurn(role="user", content=request.question))
        history.append(ConversationTurn(role="assistant", content=answer_text))
        _conversations[session_id] = history

    return EventSourceResponse(event_generator())
