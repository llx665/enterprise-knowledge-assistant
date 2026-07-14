# ============================================================
# 全局配置模块
# 负责加载 .env 配置并提供统一的配置对象
# ============================================================
import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，优先级: 环境变量 > .env 文件"""

    # LLM 配置
    llm_api_key: str = "sk-your-api-key-here"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3

    # Embedding 配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # Rerank 配置
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # Redis 配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # ChromaDB 配置
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "enterprise_kb"

    # RAG 检索参数
    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_fusion: int = 5
    rerank_top_k: int = 5
    rrf_k: int = 60
    chunk_size: int = 512
    chunk_overlap: int = 128
    parent_chunk_size: int = 2048

    # 缓存
    cache_ttl: int = 300
    cache_max_size: int = 1000

    # 记忆
    short_memory_ttl: int = 1800
    long_memory_path: str = "./data/memory.json"

    # 服务
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
