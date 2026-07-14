# 企业智能知识库助手

基于 RAG（检索增强生成）技术的企业智能知识库问答系统。
支持多格式文档解析、混合检索、流式输出、ReAct 自省校验。

## 技术栈

- 后端: Python 3.10 + FastAPI + LangChain + ChromaDB
- 检索: BM25 + 向量检索 + RRF 融合 + BGE 重排序
- 缓存: 本地内存 + Redis 双层缓存
- 异步: RQ (Redis Queue)
- LLM: OpenAI 兼容接口（支持 DeepSeek / Ollama 等）
- 前端: 原生 HTML + CSS + JavaScript
- 容器: Docker + Docker Compose

## 快速启动 (Docker)

```bash
cd enterprise-knowledge-assistant
docker-compose up -d --build
```

访问 http://localhost:8000

## 功能特性

| 模块 | 功能 |
|------|------|
| 文档解析 | PDF/Word/HTML/MD/TXT, 父子分层分块 |
| 向量索引 | ChromaDB, 增量更新, 版本管理 |
| Self-RAG | LLM 自主判断是否需要检索 |
| 混合检索 | 向量 + BM25 + RRF + BGE 重排 |
| 溯源防幻觉 | 答案绑定原文片段/文档名/页码 |
| 缓存 | 本地 + Redis 双层, 防穿透 |
| 日志 | 审计日志, 检索统计, TOP10 |
| 记忆 | 短期摘要 + 长期偏好 |
| ReAct 自省 | 自动校验, 不匹配重新检索 |
| RAG 评测 | 自动计算忠实度/精确率/召回率 |
| 异步处理 | RQ 队列, 不阻塞问答 |
| 权限控制 | 访客/管理员角色 |
| 熔断机制 | 超时熔断与重试 |
| Agent 扩展 | 预留检索 Agent/溯源校验 Agent |

## API 接口

| 路径 | 方法 | 说明 |
|------|------|------|
| /api/auth/login | POST | 登录 |
| /api/upload | POST | 上传文档 |
| /api/chat/ask | POST | SSE 流式问答 |
| /api/knowledge/documents | GET | 文档列表 |
| /api/index/info | GET | 索引信息 |
| /api/index/rebuild | POST | 重建索引 |
| /api/logs/list | GET | 日志查询 |
| /api/logs/stats | GET | 日志统计 |
| /api/eval/run | GET | RAG 评测 |
| /health | GET | 健康检查 |

## 环境配置

编辑 `.env` 文件配置：

```
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## License

MIT
