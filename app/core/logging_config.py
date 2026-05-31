"""结构化日志配置 — 基于 structlog 的 JSON 日志。

开发环境输出彩色控制台格式，生产环境输出 JSON Lines 格式，
方便接入 Elasticsearch / Datadog / Loki 等日志聚合系统。
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import settings


def setup_logging() -> None:
    """初始化结构化日志系统。

    启动时调用一次即可（在 app/main.py 的 lifespan 中调用）。

    开发模式（DEBUG=true）：
        - 彩色控制台输出，便于调试
    生产模式（DEBUG=false）：
        - JSON Lines 输出，方便日志聚合和搜索
    """
    # 两组模式共享的基础处理器链
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,     # 合并上下文变量
        structlog.stdlib.add_logger_name,            # 添加 logger 名称
        structlog.stdlib.add_log_level,              # 添加日志级别
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"), # ISO 8601 时间戳
        structlog.processors.StackInfoRenderer(),    # 堆栈信息
        structlog.processors.format_exc_info,        # 异常信息格式化
        structlog.processors.UnicodeDecoder(),       # Unicode 解码
    ]

    if settings.DEBUG:
        # 开发环境：彩色控制台输出
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # 生产环境：JSON Lines 输出
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # 静默第三方库的冗余日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("unstructured").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取指定模块的结构化日志记录器。

    Args:
        name: 模块名（通常传 __name__），为空时使用调用方模块名。

    Returns:
        绑定日志器，支持 .info("事件名", key=value) 的键值对日志。
    """
    return structlog.get_logger(name or __name__)
