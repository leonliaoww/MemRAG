# RAG Enterprise — 企业级 RAG 知识库系统

基于 **LangChain + FastAPI + Chroma** 构建的生产级 RAG（检索增强生成）知识库系统。
架构预留 LangGraph 迁移接口，支持从简单链式调用平滑升级到 Agentic RAG。

## 系统架构

```
┌──────────────────────────────────┐
│  FastAPI REST + SSE 流式输出      │  ← API 层
├──────────────────────────────────┤
│  LangChain RAG Chains            │  ← 编排层
│  （Phase 3 迁移至 LangGraph）      │
├──────────┬──────────┬───────────┤
│ 文档摄取  │ 混合检索  │ 答案生成   │  ← 流水线层
├──────────┴──────────┴───────────┤
│ Chroma 向量库 · Redis 缓存 · OpenAI│  ← 基础设施层
└──────────────────────────────────┘
```

**数据流：** 用户问题 → 向量/BM25 混合检索 → CrossEncoder 重排序 → 上下文注入 Prompt → LLM 生成答案 → 引用追踪

## 快速开始

### 环境要求

- Python 3.11+
- Docker（用于运行 Chroma + Redis）
- OpenAI API Key

### 1. 初始化

```bash
# 进入项目目录
cd rag-enterprise

# 复制环境配置
cp .env.example .env
# 编辑 .env → 填入你的 OPENAI_API_KEY

# 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"    # Windows
# 或 source .venv/bin/pip install -e ".[dev]"  # macOS/Linux

# 启动基础设施（Chroma + Redis）
make docker-up
```

### 2. 启动服务

```bash
# 开发模式（热重载，自动启用 /docs 接口文档）
make dev

# 生产模式
make run
```

浏览器打开 **http://localhost:8000/docs** 查看交互式 API 文档。

### 3. 上传文档并提问

```bash
# 上传一个文档（支持 PDF、DOCX、TXT、MD、HTML、CSV）
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@你的文档.pdf"

# 对知识库提问
curl -X POST http://localhost:8000/api/v1/queries/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "季度营收是多少？", "top_k": 5, "retrieval_mode": "hybrid"}'

# 流式问答（SSE，逐 token 返回）
curl -X POST http://localhost:8000/api/v1/queries/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "总结一下主要内容", "stream": true}'

# 多轮对话
curl -X POST http://localhost:8000/api/v1/queries/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "上一段提到的数据具体是多少？"}'
```

## API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/documents/upload` | 上传并摄取文档 |
| `GET` | `/api/v1/documents` | 列出所有文档 |
| `GET` | `/api/v1/documents/{id}` | 查询文档详情 |
| `DELETE` | `/api/v1/documents/{id}` | 删除文档及其向量 |
| `POST` | `/api/v1/queries/ask` | RAG 问答（返回完整答案+引用） |
| `POST` | `/api/v1/queries/ask/stream` | RAG 流式问答（SSE） |
| `POST` | `/api/v1/queries/chat` | 多轮对话 RAG |
| `POST` | `/api/v1/queries/chat/stream` | 多轮对话流式 RAG |
| `GET` | `/api/v1/admin/health` | 系统健康检查 |
| `GET` | `/api/v1/admin/collections` | Chroma collection 统计 |
| `POST` | `/api/v1/admin/collections/{name}` | 创建 collection |

## 检索模式

| 模式 | 原理 | 适用场景 |
|------|------|----------|
| `vector` | 向量语义检索（OpenAI embeddings 余弦相似度） | 语义匹配、模糊查询 |
| `hybrid` | 向量 + BM25 关键词，RRF 融合排序（**默认**） | 通用场景，兼顾语义和关键词 |
| `keyword` | 纯 BM25 稀疏检索 | 精确关键词匹配 |

检索管道：**粗筛**（向量/BM25，毫秒级）→ **精排**（CrossEncoder，百毫秒级）。

## 项目结构

```
rag-enterprise/
├── app/
│   ├── main.py                       # FastAPI 应用入口（lifespan/CORS/路由注册）
│   ├── api/
│   │   ├── deps.py                   # 依赖注入（配置、向量库）
│   │   └── v1/
│   │       ├── documents.py          # 文档上传/列表/删除（状态码 202）
│   │       ├── queries.py            # RAG 问答 + 多轮对话 + 流式输出
│   │       └── admin.py              # 健康检查 / collection 管理
│   ├── core/
│   │   ├── config.py                 # 集中配置（pydantic-settings，支持 .env）
│   │   ├── database.py               # 异步 SQLAlchemy 引擎（可选）
│   │   └── logging_config.py         # structlog 结构化日志（开发彩色/生产 JSON）
│   ├── models/
│   │   └── document.py               # ORM 模型：租户 → 知识库 → 文档
│   ├── schemas/
│   │   ├── common.py                 # 通用 Schema（分页/健康检查/错误）
│   │   ├── document.py               # 文档上传和状态 Schema
│   │   └── query.py                  # 问答请求/响应/引用 Schema
│   ├── engine/                       # 核心引擎（项目核心）
│   │   ├── ingestion/                # 文档摄取
│   │   │   ├── loader.py             # 加载器工厂（6 种格式）
│   │   │   ├── splitter.py           # 智能切分（含中文分隔符）
│   │   │   └── embedder.py           # OpenAI embeddings 封装
│   │   ├── retrieval/                # 检索引擎
│   │   │   ├── vector_store.py       # Chroma 向量库封装（collection 管理）
│   │   │   ├── hybrid.py             # 混合检索（向量 + BM25 + RRF 融合）
│   │   │   └── reranker.py           # CrossEncoder 重排序
│   │   └── generation/               # 生成引擎
│   │       ├── llm.py                # LLM 适配器（普通+流式，可替换供应商）
│   │       ├── prompts.py            # Prompt 模板库（RAG/对话/查询改写/评分）
│   │       └── citations.py          # 引用提取与模糊匹配
│   ├── chains/                       # LangChain 编排
│   │   ├── rag_chain.py              # 标准 RAG 链（6 阶段管道）
│   │   ├── conversational_chain.py   # 多轮对话 RAG（滑动窗口历史）
│   │   └── agents/                   # LangGraph Agent 预留目录
│   ├── services/
│   │   └── ingestion.py              # 摄取服务（单文件 + 批量目录）
│   ├── middleware/
│   │   └── error_handler.py          # 全局异常捕获 → 标准化错误响应
│   └── tasks/                        # Celery 异步任务预留目录
├── tests/
│   ├── test_api.py                   # API 层冒烟测试
│   └── test_ingestion.py             # 摄取管道集成测试
├── docker-compose.yml                # Chroma + Redis 一键启动
├── Dockerfile                        # 应用容器镜像
├── Makefile                          # 常用命令快捷方式
├── pyproject.toml                    # 项目依赖与工具配置
└── .env.example                      # 环境变量模板
```

## 版本路线

### ✅ Phase 1 — MVP（当前版本）

- [x] 多格式文档摄取（PDF / DOCX / TXT / Markdown / HTML / CSV）
- [x] 向量检索 + 混合检索（BM25 + RRF 融合）+ CrossEncoder 重排序
- [x] 标准 RAG 链 + 多轮对话链
- [x] SSE 流式输出（逐 token 推送）
- [x] 来源引用追踪（`[文件名]` 标签 → 结构化 SourceCitation）
- [x] 全中文代码注释 + 文档

### 🔲 Phase 2 — 企业特性

- [ ] 认证与授权（API Key + JWT）
- [ ] 多租户隔离（Chroma collection 级别）
- [ ] Redis 缓存（查询去重 + 结果缓存）
- [ ] Celery 异步文档处理（大文件上传不阻塞）
- [ ] Prometheus 指标 + Grafana 监控面板

### 🔲 Phase 3 — LangGraph 升级

- [ ] Agentic RAG：自适应检索（判断是否需要检索 → 改写查询 → 选择策略）
- [ ] Self-RAG：自反思链路（检索 → 评分相关性 → 决定是否补充检索）
- [ ] 多跳推理：复杂问题拆解为多步检索子问题
- [ ] 查询改写节点（指代消解 + 上下文补全）

## 配置参考

所有配置通过 `.env` 文件或环境变量设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | *必填* | OpenAI API 密钥 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 对话模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型 |
| `CHROMA_HOST` | `localhost` | Chroma 服务地址 |
| `CHROMA_PORT` | `8001` | Chroma 端口 |
| `CHUNK_SIZE` | `1000` | 文档切分大小（字符数） |
| `CHUNK_OVERLAP` | `200` | 相邻 chunk 重叠字符数 |
| `RETRIEVAL_TOP_K` | `5` | 默认召回文档数 |
| `HYBRID_SEARCH_ENABLED` | `true` | 启用混合检索 |
| `RERANK_ENABLED` | `true` | 启用 CrossEncoder 重排序 |
| `MAX_UPLOAD_SIZE_MB` | `50` | 单文件最大上传大小 |

## 开发指南

```bash
make test        # 运行测试
make lint        # Ruff 代码检查
make format      # Ruff 自动格式化
make test-cov    # 测试覆盖率报告
make typecheck   # Mypy 静态类型检查
make clean       # 清理构建产物
```
