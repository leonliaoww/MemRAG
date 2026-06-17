# RAG Enterprise — 企业级 RAG 知识库系统

基于 **LangChain + FastAPI + Chroma** 构建的生产级 RAG（检索增强生成）知识库系统。
架构预留 LangGraph 迁移接口，支持从简单链式调用平滑升级到 Agentic RAG。

## 系统架构

```
┌──────────────────────────────────┐
│  FastAPI REST + SSE 流式输出      │  ← API 层
├──────────────────────────────────┤
│  LangChain RAG Chains            │  ← 编排层
│                                  │
├──────────┬──────────┬────────────┤
│ 文档摄取  │ 混合检索  │ 答案生成   │  ← 流水线层
├──────────┴──────────┴────────────┤
│ Chroma 向量库 · BM25 索引· OpenAI │  ← 基础设施层
└──────────────────────────────────┘
```

**数据流：** 用户问题 → 查询改写 → 向量/BM25 混合检索（带低置信度二次检索） → CrossEncoder 重排序 → 上下文注入 Prompt → LLM 生成答案 → 引用追踪

## 快速开始

### 环境要求

- Python 3.11+
- Docker（用于运行 Chroma + Redis）
- OpenAI API Key

### 1. 初始化

```bash
# 进入项目目录
cd MemRAG

# 复制环境配置
cp .env.example .env
# 编辑 .env → 填入你的 OPENAI_API_KEY

# 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"    # Windows
# 或 source .venv/bin/pip install -e ".[dev]"  # macOS/Linux


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
MemRAG/
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
│   │   │   ├── loader.py             # 加载器工厂（6 种格式，支持 PyMuPDF）
│   │   │   ├── splitter.py           # 智能切分（含中文分隔符）
│   │   │   ├── embedder.py           # OpenAI embeddings 封装
│   │   │   ├── parser.py             # PDF 解析器（hi_res 策略）
│   │   │   ├── cleaner.py            # 文本清洗工具
│   │   │   ├── metadata.py           # 元数据提取
│   │   │   └── vision.py             # VLM 视觉处理（图表理解）
│   │   ├── retrieval/                # 检索引擎
│   │   │   ├── vector_store.py       # Chroma 向量库封装（collection 管理）
│   │   │   ├── hybrid.py             # 混合检索（向量 + BM25 + RRF 融合）
│   │   │   ├── reranker.py           # CrossEncoder 重排序（支持离线模型）
│   │   │   ├── bm25_index.py         # BM25 关键词索引
│   │   │   └── re_retrieval.py       # 低置信度二次检索
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
├── models/                           # 离线模型目录（重排序模型等）
├── chromadb/                         # Chroma 向量数据库持久化目录
├── test_pymupdf_loader.py            # PDF 加载器测试脚本
├── docker-compose.yml                # Chroma + Redis 一键启动
├── Dockerfile                        # 应用容器镜像
├── Makefile                          # 常用命令快捷方式
├── pyproject.toml                    # 项目依赖与工具配置
└── .env.example                      # 环境变量模板
```
