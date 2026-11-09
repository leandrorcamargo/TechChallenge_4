"""
Cria os embeddings das reviews e salva no vector store (ChromaDB).

Executar (a partir da raiz do repositorio):
    python rag/build_vector_store.py

Etapas:
  1. Carrega a Gold (load_gold.py) e junta os metadados das outras 3 tabelas.
  2. Gera 1 embedding por review a partir do document_text.
  3. Grava vetor + texto + metadados no ChromaDB (pasta rag/vector_store/).
"""

import time

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PASSAGE_PREFIX,
    VECTOR_STORE_DIR,
)
from load_gold import build_gold

BATCH_SIZE = 1000


def build_documents(gold):
    """Junta gold_reviews com os metadados de pedido, cliente e categoria.

    Usa as mesmas chaves descritas no README do grupo:
      gold_reviews.order_id -> gold_orders.order_id -> gold_customers.customer_id
      gold_reviews.order_id -> gold_products_by_order.order_id
    O join e LEFT: nenhuma review e perdida mesmo sem metadado.
    """
    docs = (
        gold["gold_reviews"]
        .merge(gold["gold_orders"], on="order_id", how="left")
        .merge(gold["gold_customers"], on="customer_id", how="left")
        .merge(gold["gold_products_by_order"], on="order_id", how="left")
    )
    assert len(docs) == len(gold["gold_reviews"]), "join duplicou reviews"
    return docs


def to_metadata(row):
    """Converte uma linha em metadados aceitos pelo ChromaDB.

    O Chroma so aceita str, int, float e bool, e nao aceita nulos.
    Por isso campos ausentes sao simplesmente omitidos daquele registro.
    """
    meta = {
        "review_id": row.review_id,
        "order_id": row.order_id,
        "review_score": int(row.review_score),
        "sentiment": row.sentiment,
        "review_year": int(row.review_year),
        "review_month": int(row.review_month),
        "text_length": int(row.text_length),
    }
    optional = {
        "order_status": row.order_status,
        "customer_state": row.customer_state,
        "customer_city": row.customer_city,
        "product_categories_pt": row.product_categories_pt,
        "product_categories_en": row.product_categories_en,
    }
    meta.update({k: str(v) for k, v in optional.items() if pd.notna(v)})

    # Atraso so faz sentido quando o pedido foi entregue
    if pd.notna(row.delivery_delay_days):
        meta["delivery_delay_days"] = int(row.delivery_delay_days)
        meta["is_late_delivery"] = bool(row.is_late_delivery)
    return meta


def main():
    start = time.time()

    print("1/3 Carregando Gold e juntando metadados...")
    docs = build_documents(build_gold())
    print(f"    {len(docs):,} documentos (1 review = 1 documento, sem chunking)")

    print(f"2/3 Gerando embeddings com {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = (PASSAGE_PREFIX + docs["document_text"]).tolist()
    # normalize_embeddings=True -> vetores de tamanho 1, entao o
    # produto escalar e igual a similaridade de cosseno
    embeddings = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    print(f"    {embeddings.shape[0]:,} vetores de {embeddings.shape[1]} dimensoes")

    print(f"3/3 Gravando no ChromaDB em {VECTOR_STORE_DIR} ...")
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    # Recria do zero a cada execucao, para o resultado ser reproduzivel
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    metadatas = [to_metadata(r) for r in docs.itertuples(index=False)]
    for i in range(0, len(docs), BATCH_SIZE):
        collection.add(
            ids=docs["review_id"].iloc[i : i + BATCH_SIZE].tolist(),
            embeddings=embeddings[i : i + BATCH_SIZE].tolist(),
            documents=docs["document_text"].iloc[i : i + BATCH_SIZE].tolist(),
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"\nPronto: {collection.count():,} reviews indexadas em {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
