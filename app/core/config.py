"""应用配置 — 基于 pydantic-settings 的集中配置管理。

所有配置项通过环境变量加载，提供合理的默认值。
密钥类配置（如 API Key）在运行时必须提供。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAG 企业知识库的集中配置类。

    配置优先级：环境变量 > .env 文件 > 默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略未定义的环境变量，防止意外注入
    )

    # ── 应用基础配置 ────────────────────────────────────
    APP_NAME: str = "RAG Enterprise"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── 服务器配置 ──────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── OpenAI 配置 ────────────────────────────────────
    OPENAI_API_KEY: SecretStr | None = Field(
        default=None, description="OpenAI API 密钥。运行时调用 LLM/Embedding 前必须设置"
    )
    OPENAI_MODEL: str = "gpt-4o-mini"           # 对话模型
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # 嵌入模型
    OPENAI_TEMPERATURE: float = 0.0              # 生成温度（0=确定性输出）
    OPENAI_MAX_TOKENS: int = 2048                # 最大生成 token 数

    # ── Chroma 向量库配置 ──────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # ── Redis 配置 ──────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── PostgreSQL（可选的关系型存储）───────────────────
    DATABASE_URL: str = ""  # 为空时不启用关系型存储

    # ── 文档处理配置 ────────────────────────────────────
    CHUNK_SIZE: int = 1000          # 文档切分块大小（字符数）
    CHUNK_OVERLAP: int = 200        # 相邻 chunk 之间的重叠字符数
    MAX_UPLOAD_SIZE_MB: int = 50    # 单文件最大上传大小
    SUPPORTED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html", ".csv"]
    )

    # ── 检索配置 ────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5        # 默认返回的文档数
    HYBRID_SEARCH_ENABLED: bool = True   # 是否启用混合检索（向量+BM25）
    RERANK_ENABLED: bool = True          # 是否启用 CrossEncoder 重排序
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── 认证配置 ────────────────────────────────────────
    AUTH_ENABLED: bool = False           # MVP 阶段默认关闭
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ── 限流配置 ────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = False      # MVP 阶段默认关闭
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Celery 异步任务配置 ─────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"


@lru_cache
def get_settings() -> Settings:
    """返回缓存的全局配置单例（使用 lru_cache 避免重复创建）。"""
    return Settings()


# 模块级单例 — 代码中直接 `from app.core.config import settings` 即可使用
settings = get_settings()
