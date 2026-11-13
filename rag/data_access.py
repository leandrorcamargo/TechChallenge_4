"""Acesso à camada Gold e montagem do corpus do RAG.

De onde a Gold vem (GOLD_SOURCE no .env):
  - "local" (padrão): roda o código original de data_preparing/ aqui mesmo,
    via retrieval/load_gold.py, a partir dos parquets Bronze do repo. Não
    precisa de credencial AWS — e já comparamos: as 4 tabelas são idênticas
    às do S3.
  - "s3": baixa os 4 parquets do bucket (aí sim precisa das credenciais).

O corpus (.cache/gold/voc_corpus.parquet) é 1 linha por review com texto,
já com os metadados de pedido, cliente e produto (joins pelas FKs da Gold).
Ele alimenta o índice BM25, a validação de filtros e as estatísticas do
contexto. Os IDs são os mesmos do vector store (id = review_id).

    python -m rag.data_access          # gera o corpus (se ainda não existir)
    python -m rag.data_access --force  # regera
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from rag.config import get_settings

GOLD_TABLES = ["gold_reviews", "gold_orders", "gold_customers", "gold_products_by_order"]
CORPUS_FILE = "voc_corpus.parquet"
NO_CATEGORY = "sem_categoria"

CORPUS_COLUMNS = [
    "review_id", "order_id", "document_text", "text_length",
    "review_score", "sentiment", "review_year", "review_month", "review_creation_date",
    "order_status", "is_late_delivery", "delivery_delay_days",
    "customer_state", "product_categories_pt", "product_category_main",
]


# ----------------------------------------------------------------------
# Fontes da Gold
# ----------------------------------------------------------------------
def gold_from_local() -> dict[str, pd.DataFrame]:
    """Bronze do repo -> Silver -> Gold, com o loader da etapa de Embeddings."""
    from rag.vectorstore_adapter import _import_retrieval

    return _import_retrieval("load_gold").build_gold()


def gold_from_s3() -> dict[str, pd.DataFrame]:
    import boto3

    s = get_settings()
    client = boto3.client("s3", region_name=s.aws_region)  # credenciais via .env / ambiente
    tables = {}
    for name in GOLD_TABLES:
        body = client.get_object(Bucket=s.s3_bucket, Key=f"{s.gold_prefix}{name}.parquet")["Body"].read()
        tables[name] = pd.read_parquet(io.BytesIO(body))
        print(f"  [S3] {name}: {len(tables[name]):,} linhas")
    return tables


def load_gold(source: str | None = None) -> dict[str, pd.DataFrame]:
    source = (source or get_settings().gold_source).lower()
    if source == "s3":
        return gold_from_s3()
    if source == "local":
        return gold_from_local()
    raise ValueError(f"GOLD_SOURCE inválido: {source!r} (use 'local' ou 's3')")


# ----------------------------------------------------------------------
# Corpus enriquecido
# ----------------------------------------------------------------------
def build_corpus(gold: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = (
        gold["gold_reviews"].merge(gold["gold_orders"], on="order_id", how="left")
        .merge(gold["gold_customers"], on="customer_id", how="left")
        .merge(gold["gold_products_by_order"][["order_id", "product_categories_pt"]], on="order_id", how="left")
    )
    df["product_categories_pt"] = df["product_categories_pt"].fillna(NO_CATEGORY)
    # a 1ª da lista vira a categoria "principal" (quase todo pedido tem uma só)
    df["product_category_main"] = df["product_categories_pt"].str.split(", ").str[0]
    df["is_late_delivery"] = df["is_late_delivery"].fillna(False).astype(bool)
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"]).dt.strftime("%Y-%m-%d")
    for col in ("review_score", "review_year", "review_month", "text_length"):
        df[col] = df[col].astype(int)
    return df[CORPUS_COLUMNS].reset_index(drop=True)


def corpus_path():
    return get_settings().gold_cache_dir / CORPUS_FILE


def ensure_corpus(force: bool = False) -> pd.DataFrame:
    path = corpus_path()
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = build_corpus(load_gold())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


@lru_cache(maxsize=1)
def load_corpus() -> pd.DataFrame:
    """Corpus em memória (lido do cache; gerado na primeira vez)."""
    return ensure_corpus()


# ----------------------------------------------------------------------
# Vocabulário controlado (para validar filtros extraídos pelo LLM)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Vocabulary:
    categories: tuple[str, ...]
    states: tuple[str, ...]
    years: tuple[int, ...]
    order_statuses: tuple[str, ...]


@lru_cache(maxsize=1)
def load_vocabulary() -> Vocabulary:
    df = load_corpus()
    cats = sorted({c for s in df["product_categories_pt"] for c in s.split(", ")} - {NO_CATEGORY})
    return Vocabulary(
        categories=tuple(cats),
        states=tuple(sorted(df["customer_state"].dropna().unique())),
        years=tuple(sorted(int(y) for y in df["review_year"].unique())),
        order_statuses=tuple(sorted(df["order_status"].dropna().unique())),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gera o corpus do RAG a partir da Gold")
    p.add_argument("--force", action="store_true", help="regera o corpus")
    p.add_argument("--source", choices=["local", "s3"], default=None, help="fonte da Gold (padrão: GOLD_SOURCE)")
    a = p.parse_args(argv)

    s = get_settings()
    source = a.source or s.gold_source
    print(f"Gold ({source}) -> corpus em {corpus_path()}")
    if a.source:
        df = build_corpus(load_gold(a.source))
        corpus_path().parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(corpus_path(), index=False)
    else:
        df = ensure_corpus(force=a.force)
    print(f"\nCorpus: {len(df):,} reviews | {df['product_category_main'].nunique()} categorias | "
          f"{df['customer_state'].nunique()} UFs | anos {sorted(int(y) for y in df['review_year'].unique())}")
    print(df["sentiment"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
