"""Fixtures: corpus mínimo em memória + dublês de LLM, busca densa e reranker.
Os testes não precisam de Ollama, S3, índice vetorial nem download de modelos."""

import pandas as pd
import pytest

from rag.data_access import CORPUS_COLUMNS, Vocabulary

ROWS = [
    # review_id, texto, nota, sentimento, categoria, UF, atrasou
    ("r1", "Produto não chegou, já passou do prazo de entrega", 1, "negative", "cama_mesa_banho", "SP", True),
    ("r2", "Entrega atrasou muito, demorou mais de um mês", 1, "negative", "beleza_saude", "RJ", True),
    ("r3", "Veio quebrado e a embalagem estava danificada", 2, "negative", "utilidades_domesticas", "SP", False),
    ("r4", "Ótimo produto, chegou antes do prazo", 5, "positive", "cama_mesa_banho", "MG", False),
    ("r5", "Ótimo produto", 5, "positive", "beleza_saude", "SP", False),
    ("r6", "ótimo produto!", 5, "positive", "beleza_saude", "SP", False),
    ("r7", "Não recebi o produto até agora, péssimo", 1, "negative", "informatica_acessorios", "BA", True),
    ("r8", "Produto razoável, mas a cor é diferente da foto", 3, "neutral", "moveis_decoracao", "PR", False),
]


@pytest.fixture
def corpus() -> pd.DataFrame:
    data = []
    for rid, text, score, sent, cat, uf, late in ROWS:
        data.append({
            "review_id": rid, "order_id": "o" + rid, "document_text": text, "text_length": len(text),
            "review_score": score, "sentiment": sent, "review_year": 2018, "review_month": 3,
            "review_creation_date": "2018-03-01", "order_status": "delivered", "is_late_delivery": late,
            "delivery_delay_days": 5.0 if late else -3.0, "customer_state": uf,
            "product_categories_pt": cat, "product_category_main": cat,
        })
    return pd.DataFrame(data)[CORPUS_COLUMNS]


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary(
        categories=("beleza_saude", "cama_mesa_banho", "informatica_acessorios", "moveis_decoracao",
                    "utilidades_domesticas"),
        states=("BA", "MG", "PR", "RJ", "SP"),
        years=(2016, 2017, 2018),
        order_statuses=("canceled", "delivered", "shipped"),
    )


@pytest.fixture
def patched(monkeypatch, corpus, vocab):
    """Substitui vocabulário e índice BM25 globais pelos do corpus de teste."""
    import rag.query_processing as qp
    import rag.retrieval as rt

    monkeypatch.setattr(qp, "load_vocabulary", lambda: vocab)
    index = rt.BM25Index(corpus)
    monkeypatch.setattr(rt, "get_bm25_index", lambda: index)
    return corpus
