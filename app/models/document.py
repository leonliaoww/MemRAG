"""SQLAlchemy 数据模型 — 租户/知识库/文档三层结构。

租户 → 知识库（1:N）
知识库 → 文档（1:N）

多租户隔离通过 Chroma collection 名称实现，
关系型数据库存储元数据（文件名、状态、时间戳等）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Tenant(Base):
    """多租户隔离单元。

    每个租户拥有一个或多个知识库。
    单租户部署时，仅使用一条 'default' 租户记录即可。
    """

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    api_key_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
        comment="API 密钥的 bcrypt 哈希值"
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 级联删除：删除租户时同时删除其所有知识库
    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )


class KnowledgeBase(Base):
    """知识库 — 租户下的文档集合。

    每个知识库对应 Chroma 中的一个独立 collection，
    实现检索级别的租户隔离。
    """

    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Chroma collection 名称 — 用于向量检索隔离
    chroma_collection: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="knowledge_bases")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    """文档元数据 — 记录每个已摄入文档的处理状态和统计信息。

    实际的文档内容以 chunk 形式存储在 Chroma 向量库中，
    此表仅记录元数据（文件名、大小、状态、chunk 数量等）。
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # 文档处理状态：pending → processing → completed / failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 额外元数据的 JSON 字符串（灵活扩展）
    metadata_: Mapped[str | None] = mapped_column(
        "metadata", Text, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
