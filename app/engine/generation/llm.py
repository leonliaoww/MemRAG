"""LLM 适配器 — 多供应商对话模型封装。

支持供应商：
    - 阿里云 DashScope（通义千问）：qwen-plus / qwen-max / qwen-turbo
    - OpenAI：gpt-4o-mini / gpt-4o / gpt-3.5-turbo

通过 LLM_PROVIDER 配置选择供应商。
适配器接口统一（generate / generate_stream），上层 chain 代码无需感知供应商差异。

DashScope 模型选择建议：
    qwen-plus      — 性价比最优，适合大多数 RAG 场景（默认）
    qwen-max       — 最强推理能力，适合复杂分析
    qwen-turbo     — 最快响应，适合简单问答
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 单例 LLM（非流式）
_llm: BaseChatModel | None = None


def _create_dashscope_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """创建阿里云 DashScope（通义千问）对话模型。

    使用 langchain_community 的 ChatTongyi 封装。
    """
    from langchain_community.chat_models.tongyi import ChatTongyi

    if settings.DASHSCOPE_API_KEY is None:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未设置。请在 .env 中配置阿里云 DashScope API Key"
        )

    return ChatTongyi(
        model=model or settings.DASHSCOPE_LLM_MODEL,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        dashscope_api_key=settings.DASHSCOPE_API_KEY.get_secret_value(),
        streaming=streaming,
    )


def _create_openai_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """创建 OpenAI 对话模型。"""
    from langchain_openai import ChatOpenAI

    if settings.OPENAI_API_KEY is None:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置。请在 .env 中配置 OpenAI API Key"
        )

    return ChatOpenAI(
        model=model or settings.OPENAI_MODEL,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        streaming=streaming,
    )


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """创建配置好的对话模型（根据 LLM_PROVIDER 自动选择供应商）。

    Args:
        model: 模型名称（覆盖供应商默认值）。
        temperature: 生成温度。
        max_tokens: 最大输出 token 数。
        streaming: 是否启用流式输出。

    Returns:
        LangChain 兼容的 ChatModel 实例。
    """
    if settings.LLM_PROVIDER == "dashscope":
        return _create_dashscope_llm(model, temperature, max_tokens, streaming)
    else:
        return _create_openai_llm(model, temperature, max_tokens, streaming)


def get_singleton_llm() -> BaseChatModel:
    """获取缓存的单例 LLM（默认非流式）。

    全局复用同一个实例和 HTTP 连接池。
    """
    global _llm
    if _llm is None:
        _llm = get_llm()
        model_name = (
            settings.DASHSCOPE_LLM_MODEL
            if settings.LLM_PROVIDER == "dashscope"
            else settings.OPENAI_MODEL
        )
        logger.info(
            "LLM 初始化完成",
            provider=settings.LLM_PROVIDER,
            model=model_name,
        )
    return _llm


async def generate(
    messages: list[BaseMessage],
    model: str | None = None,
) -> str:
    """生成单次文本补全（非流式）。

    Args:
        messages: LangChain 消息列表。
        model: 可选的模型覆盖。

    Returns:
        模型生成的完整文本。
    """
    llm = get_llm(model=model) if model else get_singleton_llm()
    result = await llm.ainvoke(messages)
    content = result.content
    if isinstance(content, list):
        return str(content[0]) if content else ""
    return str(content)


async def generate_stream(
    messages: list[BaseMessage],
    model: str | None = None,
):
    """流式生成 — 逐个 token 异步产出。

    Args:
        messages: LangChain 消息列表。
        model: 可选的模型覆盖。

    Yields:
        字符串 token 片段。
    """
    llm = get_llm(model=model, streaming=True) if model else get_llm(streaming=True)
    async for chunk in llm.astream(messages):
        content = chunk.content
        if content:
            yield str(content)
