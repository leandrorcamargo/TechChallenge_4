"""Config do RAG, lida do .env na raiz do projeto.

Tudo que o projeto baixa ou gera (Ollama portátil, modelos, caches) fica
dentro de .tools/ e .cache/, que estão no .gitignore. Assim não suja a
máquina de ninguém: pra "desinstalar" é só apagar as duas pastas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / ".tools"
CACHE_DIR = PROJECT_ROOT / ".cache"

load_dotenv(PROJECT_ROOT / ".env")

# cache do HF dentro do projeto, senão espalha arquivo no home do usuário
os.environ.setdefault("HF_HOME", str(TOOLS_DIR / "hf"))
# telemetria desligada
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# no Windows sem modo desenvolvedor o HF não consegue criar symlink;
# funciona igual, só solta um aviso chato
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- Ollama / LLM ---
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen3.8"))
    ollama_model_analyzer: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL_ANALYZER", _env("OLLAMA_MODEL", "qwen3.8"))
    )
    ollama_version: str = field(default_factory=lambda: _env("OLLAMA_VERSION", "v0.40.15"))
    ollama_dir: Path = field(default_factory=lambda: _path("OLLAMA_DIR", TOOLS_DIR / "ollama"))
    ollama_models_dir: Path = field(
        default_factory=lambda: _path("OLLAMA_MODELS_DIR", TOOLS_DIR / "ollama" / "models")
    )
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_num_ctx: int = field(default_factory=lambda: int(_env("LLM_NUM_CTX", "8192")))

    # --- Dados (Gold) ---
    # "local": Gold gerada do Bronze do repo (retrieval/load_gold.py), sem AWS. "s3": bucket.
    gold_source: str = field(default_factory=lambda: _env("GOLD_SOURCE", "local"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "amzn-s3-fiap-tech-4"))
    gold_prefix: str = field(default_factory=lambda: _env("GOLD_PREFIX", "db_gold/"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-2"))
    gold_cache_dir: Path = field(default_factory=lambda: _path("GOLD_CACHE_DIR", CACHE_DIR / "gold"))

    # Vector store e modelo de embedding: definidos pela etapa de Embeddings em
    # retrieval/config.py (ChromaDB em retrieval/vector_store, multilingual-e5-small).

    # --- Recuperação / seleção ---
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )
    k_dense: int = field(default_factory=lambda: int(_env("K_DENSE", "40")))
    k_bm25: int = field(default_factory=lambda: int(_env("K_BM25", "40")))
    k_fused: int = field(default_factory=lambda: int(_env("K_FUSED", "50")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    top_n: int = field(default_factory=lambda: int(_env("TOP_N", "12")))
    rerank_threshold: float = field(default_factory=lambda: float(_env("RERANK_THRESHOLD", "-6.2")))
    min_evidence: int = field(default_factory=lambda: int(_env("MIN_EVIDENCE", "3")))


def get_settings() -> Settings:
    return Settings()
