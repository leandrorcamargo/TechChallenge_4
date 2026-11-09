"""
Busca semantica no vector store — a etapa de RECUPERACAO do RAG.

Uso como modulo (para quem for montar o contexto e chamar o LLM):
    from search import search
    resultados = search("problemas com entrega", k=5, where={"sentiment": "negative"})

Uso pela linha de comando (teste rapido):
    python retrieval/search.py "o produto chegou quebrado"
"""

import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL, QUERY_PREFIX, VECTOR_STORE_DIR


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_collection(COLLECTION_NAME)


def search(question, k=5, where=None, min_similarity=None):
    """Retorna as k reviews mais parecidas com a pergunta.

    question        texto da pergunta do usuario
    k               quantidade de reviews a retornar
    where           filtro opcional por metadados (sintaxe do ChromaDB), ex.:
                      {"sentiment": "negative"}
                      {"review_score": {"$lte": 2}}
                      {"$and": [{"customer_state": "SP"}, {"is_late_delivery": True}]}
    min_similarity  se informado, descarta resultados abaixo desse valor
                    (util para detectar perguntas sem evidencia suficiente)

    Cada resultado e um dict com review_id, similarity (0 a 1), text e metadata.
    """
    # 1. Transformacao da pergunta: prefixo "query: " + embedding normalizado
    query_vec = _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)

    # 2. Recuperacao: vizinhos mais proximos por cosseno no ChromaDB
    res = _collection().query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        where=where,
    )

    results = []
    for rid, dist, text, meta in zip(
        res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
    ):
        similarity = 1 - dist  # distancia de cosseno -> similaridade
        if min_similarity is None or similarity >= min_similarity:
            results.append(
                {"review_id": rid, "similarity": round(similarity, 3), "text": text, "metadata": meta}
            )
    return results


def print_results(question, results):
    print(f"\nPergunta: {question}")
    if not results:
        print("  (nenhuma review relevante encontrada)")
    for r in results:
        m = r["metadata"]
        print(
            f"  [{r['similarity']:.3f}] nota {m['review_score']} | "
            f"{m.get('product_categories_pt', '-')} | {m.get('customer_state', '-')} | "
            f"id {r['review_id'][:8]}\n      {r['text'][:120]}"
        )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "o produto chegou quebrado"
    print_results(q, search(q))
