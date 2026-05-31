"""LLM 适配器 — OpenAI Chat 模型的轻量封装。

设计原则：
    - 统一接口：切换模型供应商（Anthropic / 本地 Ollama / Azure OpenAI）
      只需修改此文件，chain 层代码无需变动
    - 流式支持：同一套消息结构支持普通生成和 token 级流式生成
    - 单例缓存：复用 ChatOpenAI 实例及其 HTTP 连接池

未来扩展：
    - 支持 Anthropic Claude（消息格式转换）
    - 支持本地 vLLM 端点（OpenAI 兼容 API）
    - 支持 Azure OpenAI（认证方式不同）
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 单例 LLM（非流式）
_llm: BaseChatModel | None = None


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """创建配置好的 ChatOpenAI 实例。

    Args:
        model: 模型名称（覆盖配置默认值）。
        temperature: 生成温度（0=确定性，1=最大随机性）。
        max_tokens: 最大输出 token 数。
        streaming: 是否启用 token 级流式输出。

    Returns:
        LangChain 兼容的 ChatModel 实例。
    """
    if settings.OPENAI_API_KEY is None:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置。请在 .env 文件中配置 OPENAI_API_KEY"
        )
    return ChatOpenAI(
        model=model or settings.OPENAI_MODEL,
        temperature=temperature if temperature is not None else settings.OPENAI_TEMPERATURE,
        max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        streaming=streaming,
    )


def get_singleton_llm() -> BaseChatModel:
    """获取缓存的单例 LLM（默认非流式）。

    全局复用同一个 ChatOpenAI 实例和 HTTP 连接池。
    """
    global _llm
    if _llm is None:
        _llm = get_llm()
        logger.info("LLM 初始化完成", model=settings.OPENAI_MODEL)
    return _llm


async def generate(
    messages: list[BaseMessage],
    model: str | None = None,
) -> str:
    """生成单次文本补全（非流式）。

    Args:
        messages: LangChain 消息列表（SystemMessage + HumanMessage 等）。
        model: 可选的模型覆盖。

    Returns:
        模型生成的完整文本。
    """
    llm = get_llm(model=model) if model else get_singleton_llm()
    result = await llm.ainvoke(messages)
    content = result.content
    # 处理多模态返回（RAG 场景极少出现，但做安全兜底）
    if isinstance(content, list):
        return str(content[0]) if content else ""
    return str(content)


async def generate_stream(
    messages: list[BaseMessage],
    model: str | None = None,
):
    """流式生成 — 逐个 token 异步产出。

    使用 async for 遍历，每个 yield 一个 token 字符串。

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
