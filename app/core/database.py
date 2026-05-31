"""异步 SQLAlchemy 引擎和会话工厂。

使用 asyncpg 驱动连接 PostgreSQL。
当 DATABASE_URL 为空时（默认），关系型存储不启用，仅使用 Chroma 向量库。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。"""
    pass


# 当 DATABASE_URL 未配置时，engine 和 session 工厂为 None
# 此时系统仅使用 Chroma 向量库，不启用关系型存储
engine = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None

if settings.DATABASE_URL:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,         # DEBUG 下打印 SQL
        pool_size=20,                # 连接池大小
        max_overflow=10,             # 最大溢出连接数
        pool_pre_ping=True,          # 连接前 ping 检测有效性
    )
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,      # 提交后不使对象过期
    )


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI 依赖注入：提供一个异步数据库会话。

    请求开始时创建会话，结束时自动提交或回滚。

    Raises:
        RuntimeError: 当 DATABASE_URL 未配置时抛出。
    """
    if async_session_factory is None:
        raise RuntimeError(
            "DATABASE_URL 未配置。请在 .env 中设置 DATABASE_URL 以启用关系型存储。"
        )
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """启动时创建所有数据库表。未配置数据库时无操作。"""
    if engine is not None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """关闭应用时释放数据库连接池。"""
    if engine is not None:
        await engine.dispose()
