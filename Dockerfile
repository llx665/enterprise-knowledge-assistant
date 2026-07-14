# ============================================================
# 企业智能知识库助手 - Docker 构建文件
# ============================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- 运行阶段 ----------
FROM python:3.10-slim

WORKDIR /app

# 拷贝已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 系统运行时依赖（供 pypdf/lxml 等使用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 复制项目代码
COPY backend/ /app/backend/
COPY .env /app/.env

# 创建数据目录
RUN mkdir -p /app/data/chroma /app/data/documents /app/data/logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
