"""Integração com a etapa de Embeddings e Vector Store (pasta `retrieval/`).

O vector store é o da parte de Embeddings (retrieval/):
  - ChromaDB persistente em `retrieval/vector_store/`, coleção `olist_reviews`;
  - embeddings `intfloat/multilingual-e5-small` com prefixos `query:`/`passage:`;
  - 1 review = 1 documento, id = review_id, metadados de pedido/cliente/produto.

O RAG consome a função `search()` de `retrieval/search.py` como a busca densa
do 1º estágio, traduzindo os filtros extraídos da pergunta para a sintaxe
`where` do Chroma (ver `rag.filters.to_chroma_where`).

Os scripts de `retrieval/` usam imports "planos" (`from config import ...`),
então a pasta é adicionada ao sys.path antes de importá-los. `rag.config` é
importado antes para que o cache HuggingFace fique em `.tools/hf` (isolamento).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from types import ModuleType

import pandas as pd

from rag.config import PROJECT_ROOT
from rag.filters import to_chroma_where
from rag.schemas import Filters

RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"


def _import_retrieval(module: str) -> ModuleType:
    path = str(RETRIEVAL_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    # Os scripts de retrieval/ usam nomes genéricos (config, search): garante que o
    # módulo carregado é o da pasta, e não outro homônimo já importado no processo.
    for name in ("config", module):
        loaded = sys.modules.get(name)
        origin = getattr(loaded, "__file__", None) or ""
        if loaded is not None and not origin.startswith(path):
            raise ImportError(f"Conflito de nome: módulo '{name}' já importado de {origin or '?'}, "
                              f"esperado em {path}")
    return __import__(module)


def retrieval_config() -> ModuleType:
    """Constantes da etapa de Embeddings (modelo, prefixos, caminho, coleção)."""
    return _import_retrieval("config")


def vector_store_count() -> int | None:
    """Nº de documentos no vector store, ou None se ele ainda não foi construído."""
    import chromadb

    cfg = retrieval_config()
    if not cfg.VECTOR_STORE_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(cfg.VECTOR_STORE_DIR))
    if cfg.COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        return None
    return client.get_collection(cfg.COLLECTION_NAME).count()


@lru_cache(maxsize=1)
def _search_fn():
    if vector_store_count() is None:
        raise RuntimeError(
            "Vector store não encontrado em retrieval/vector_store/. "
            "Construa com: python -m rag.vectorstore_adapter build"
        )
    return _import_retrieval("search").search


def warmup() -> None:
    """Carrega o modelo de embedding e abre a coleção (usado pela interface)."""
    _search_fn()("aquecimento", k=1)


def dense_search(query: str, k: int, filters: Filters, corpus: pd.DataFrame | None = None) -> list[tuple[str, float]]:
    """Busca semântica no vector store. Retorna [(review_id, similaridade_cosseno)]."""
    results = _search_fn()(query, k=k, where=to_chroma_where(filters))
    return [(r["review_id"], float(r["similarity"])) for r in results]


def build_vector_store() -> None:
    """Executa o build da etapa de Embeddings (retrieval/build_vector_store.py)."""
    _import_retrieval("build_vector_store").main()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Vector store da etapa de Embeddings (retrieval/)")
    p.add_argument("cmd", choices=["build", "status"])
    a = p.parse_args()
    if a.cmd == "build":
        build_vector_store()
    else:
        n = vector_store_count()
        print(f"vector store: {'não construído' if n is None else f'{n:,} documentos'}")
