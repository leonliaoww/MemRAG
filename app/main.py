"""FastAPI 应用入口。

启动方式：
    make dev          # 开发模式（热重载）
    make run          # 生产模式
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging
from app.middleware.error_handler import ErrorHandlingMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理 — 启动和关闭钩子。

    启动时：
        - 初始化结构化日志
        - 连接外部服务（Chroma 客户端在首次请求时懒初始化）

    关闭时：
        - 清理资源（数据库连接池等）
    """
    # 启动
    setup_logging()
    logger.info(
        "应用启动中",
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )
    yield
    # 关闭
    logger.info("应用正在关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    包含：
        - CORS 中间件（生产环境应限制允许的源）
        - 全局错误处理中间件
        - 路由注册
        - API 文档（开发模式自动启用 Swagger 和 ReDoc）
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="企业级 RAG 知识库系统 — 基于 LangChain + Chroma + FastAPI",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── 中间件注册（洋葱模型，先注册的先执行外层）───────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ErrorHandlingMiddleware)

    # ── 路由注册 ──────────────────────────────────────
    app.include_router(api_router)

    return app


# 模块级 app 实例 — uvicorn 直接引用此对象
app = create_app()
