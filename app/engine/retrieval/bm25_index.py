"""持久化 BM25 索引 — 文件持久化、增量更新、启动加载。

功能：
    - 首次构建后保存为 pickle 文件到 BM25_INDEX_DIR
    - 后续文档摄取时增量更新（append 到现有索引）
    - 应用启动时自动加载已有索引
    - 支持按 collection 隔离索引文件

设计说明：
    BM25 索引本身是轻量级的（仅存储词频统计），适合 pickle 持久化。
    对于超大规模（100万+ 文档），建议迁移到 Elasticsearch 或外部 BM25 服务。
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 默认索引存储目录（在项目根目录下）
BM25_INDEX_DIR = Path("./bm25_index")


class PersistentBM25Index:
    """持久化 BM25 索引管理器。

    使用方式：
        index = PersistentBM25Index("my_collection")
        index.add_documents(docs)          # 构建或增量更新索引
        results = index.search("查询", 5)   # 搜索
        index.save()                       # 手动持久化
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self._index: BM25Okapi | None = None
        self._documents: list[LCDocument] = []
        self._index_file = BM25_INDEX_DIR / f"{collection_name}.pkl"

        # 启动时自动加载已有索引
        self._load()

    # ── 公开接口 ─────────────────────────────────────

    def add_documents(self, documents: list[LCDocument]) -> None:
        """增量添加文档到 BM25 索引。

        如果是首次添加，创建新索引；
        如果已有索引，将新文档追加到现有索引（需重建，BM25 不支持增量词频更新）。

        Args:
            documents: 新的文档 chunk 列表。
        """
        if not documents:
            return

        # 追加到文档列表
        self._documents.extend(documents)

        # 重建索引（包含新增文档）
        self._rebuild()

        logger.info(
            "BM25 索引已更新",
            collection=self.collection_name,
            total_docs=len(self._documents),
            added=len(documents),
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[LCDocument, float]]:
        """BM25 搜索。

        Args:
            query: 查询字符串。
            top_k: 返回数量。

        Returns:
            (文档, BM25 分数) 的排序列表，按分数降序。
        """
        if not self._index or not self._documents:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        # 排序取 top_k
        scored = sorted(
            zip(self._documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [(doc, float(score)) for doc, score in scored if score > 0]

    def save(self) -> None:
        """将索引持久化到磁盘。"""
        BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "collection_name": self.collection_name,
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in self._documents
            ],
        }

        with open(self._index_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "BM25 索引已持久化",
            collection=self.collection_name,
            docs=len(self._documents),
            file=str(self._index_file),
        )

    def delete_documents(self, doc_ids: list[str]) -> int:
        """按 ID 从索引中删除文档（需重建）。

        当前 BM25 实现要求全量重建索引，
        ChatGPT 也在用的方案。对于频繁删改的场景建议上 Elasticsearch。

        Args:
            doc_ids: 要删除的文档 ID 列表（对应 metadata.id）。

        Returns:
            实际删除的文档数。
        """
        id_set = set(doc_ids)
        before = len(self._documents)
        self._documents = [
            d for d in self._documents
            if d.metadata.get("id", "") not in id_set
        ]
        removed = before - len(self._documents)

        if removed:
            self._rebuild()

        logger.info(
            "BM25 索引文档已删除",
            collection=self.collection_name,
            removed=removed,
            remaining=len(self._documents),
        )
        return removed

    @property
    def doc_count(self) -> int:
        """索引中的文档总数。"""
        return len(self._documents)

    # ── 内部方法 ─────────────────────────────────────

    def _rebuild(self) -> None:
        """重建 BM25 索引。"""
        if not self._documents:
            self._index = None
            return

        tokenized = [doc.page_content.lower().split() for doc in self._documents]
        self._index = BM25Okapi(tokenized)

    def _load(self) -> None:
        """从磁盘加载已有索引。"""
        if not self._index_file.exists():
            logger.debug("BM25 索引文件不存在，将创建新索引", collection=self.collection_name)
            return

        try:
            with open(self._index_file, "rb") as f:
                data = pickle.load(f)

            self._documents = [
                LCDocument(
                    page_content=d["page_content"],
                    metadata=d.get("metadata", {}),
                )
                for d in data["documents"]
            ]

            self._rebuild()

            logger.info(
                "BM25 索引已从磁盘加载",
                collection=self.collection_name,
                docs=len(self._documents),
            )
        except Exception as exc:
            logger.error("BM25 索引加载失败，将重建", error=str(exc))
            self._documents = []
            self._index = None


# ── 全局索引缓存（按 collection 隔离）────────────────

_index_cache: dict[str, PersistentBM25Index] = {}


def get_bm25_index(collection_name: str | None = None) -> PersistentBM25Index:
    """获取指定 collection 的 BM25 索引（单例缓存）。

    Args:
        collection_name: Chroma collection 名称。默认使用配置中的 collection。

    Returns:
        PersistentBM25Index 实例。
    """
    name = collection_name or settings.CHROMA_COLLECTION_NAME

    if name not in _index_cache:
        _index_cache[name] = PersistentBM25Index(name)

    return _index_cache[name]
