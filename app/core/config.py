"""应用配置 — 基于 pydantic-settings 的集中配置管理。

支持多 LLM 供应商切换（DashScope / OpenAI），
通过 LLM_PROVIDER 环境变量控制，运行时自动选择对应适配器。

所有配置项通过环境变量加载，提供合理的默认值。
密钥类配置（如 API Key）在运行时必须提供。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM 供应商类型
LLMProvider = Literal["dashscope", "openai"]


class Settings(BaseSettings):
    """RAG 企业知识库的集中配置类。

    配置优先级：环境变量 > .env 文件 > 默认值

    供应商切换：
        设置 LLM_PROVIDER=dashscope → 使用阿里云 DashScope（text-embedding-v3 + qwen-plus）
        设置 LLM_PROVIDER=openai    → 使用 OpenAI（text-embedding-3-small + gpt-4o-mini）
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 应用基础配置 ────────────────────────────────────
    APP_NAME: str = "RAG Enterprise"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── 服务器配置 ──────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM 供应商选择 ────────────────────────────────
    LLM_PROVIDER: LLMProvider = "dashscope"
    """LLM 供应商：dashscope（阿里云）或 openai。切换后自动使用对应适配器。"""

    # ── 阿里云 DashScope 配置 ─────────────────────────
    DASHSCOPE_API_KEY: SecretStr | None = Field(
        default=None, description="阿里云 DashScope API Key。LLM_PROVIDER=dashscope 时必填"
    )
    DASHSCOPE_LLM_MODEL: str = "qwen-plus"
    """DashScope 对话模型：qwen-plus（性价比）/ qwen-max（最强）/ qwen-turbo（最快）"""
    DASHSCOPE_EMBEDDING_MODEL: str = "text-embedding-v3"
    """DashScope 嵌入模型：text-embedding-v3（1024维，中英文优化）"""

    # ── OpenAI 配置（LLM_PROVIDER=openai 时使用）────────
    OPENAI_API_KEY: SecretStr | None = Field(
        default=None, description="OpenAI API Key。LLM_PROVIDER=openai 时必填"
    )
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── LLM 通用参数 ───────────────────────────────────
    LLM_TEMPERATURE: float = 0.0
    """生成温度：0=确定性输出，1=最大随机性。RAG 场景建议 0。"""
    LLM_MAX_TOKENS: int = 2048

    # ── Chroma 向量库（本地持久化模式）─────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    """Chroma 本地持久化目录。向量数据以 SQLite3 + Parquet 格式存储在此目录。"""
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    """默认 collection 名称。"""

    # ── 文档处理配置 ────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html", ".csv"]
    )

    # ── 检索配置 ────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    HYBRID_SEARCH_ENABLED: bool = True
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """重排序模型：HuggingFace 模型名（联网下载）或本地路径（离线加载）。
    当 RERANK_MODEL_PATH 不为空时，优先从本地路径加载。"""
    RERANK_MODEL_PATH: str = "./models/reranker/ms-marco-MiniLM-L-6-v2"
    """重排序模型本地路径。设置后不从 HuggingFace 下载，直接从本地目录加载。"""

    # ── 数据库（可选的关系型存储）───────────────────────
    DATABASE_URL: str = ""

    # ── 认证配置 ────────────────────────────────────────
    AUTH_ENABLED: bool = False
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60


@lru_cache
def get_settings() -> Settings:
    """返回缓存的全局配置单例。"""
    return Settings()


# 模块级单例
settings = get_settings()
