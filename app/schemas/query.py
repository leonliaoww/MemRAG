"""查询/问答相关 Pydantic Schema — RAG 管道请求和响应。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    """检索策略枚举。

    - vector:  纯向量语义检索（OpenAI embeddings 余弦相似度）
    - hybrid:  混合检索 — 向量 + BM25 关键词，通过 RRF 融合排序
    - keyword: 纯 BM25 关键词检索（稀疏检索）
    """
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class SourceCitation(BaseModel):
    """答案中的单条来源引用 — 包含文档信息和相关度。"""
    document_id: str
    filename: str
    chunk_index: int
    content_preview: str = Field(
        ..., max_length=300, description="来源 chunk 的前 300 个字符"
    )
    relevance_score: float | None = None  # 相关度分数（由 CrossEncoder 或 RRF 给出）


class AskRequest(BaseModel):
    """RAG 问答请求体。

    核心字段是 question，其余字段均有合理默认值。
    """
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    knowledge_base_id: str | None = Field(
        default=None, description="可选的知识库 ID；不传时使用默认知识库"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="检索返回的文档数量")
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID  # 默认混合检索
    include_sources: bool = True    # 是否在答案中包含引用来源
    stream: bool = False            # 是否启用 SSE 流式输出


class AskResponse(BaseModel):
    """非流式 RAG 问答响应。"""
    answer: str                                # 生成的答案文本
    sources: list[SourceCitation] = Field(default_factory=list)  # 引用来源列表
    retrieval_mode: RetrievalMode              # 实际使用的检索模式
    tokens_used: int | None = None             # 消耗的 token 数（用于监控和计费）


class StreamChunk(BaseModel):
    """SSE 流式响应中的单个数据块。

    流式输出时，每个 chunk 包含一段 token 文本；
    最后一个 chunk 的 done=True，同时附带完整的 sources。
    """
    content: str
    done: bool = False
    sources: list[SourceCitation] | None = None  # 仅最后一个 chunk 非空
