"""Filtros de metadados: validação contra o vocabulário real e tradução para
(a) máscara pandas (BM25 e estatísticas) e (b) cláusula `where` do Chroma,
usando os nomes de metadados gravados pela etapa de Embeddings (retrieval/).
"""

from __future__ import annotations

import unicodedata
from typing import Any

import pandas as pd

from rag.data_access import Vocabulary
from rag.schemas import Filters

SCORES_BY_SENTIMENT = {"negative": (1, 2), "neutral": (3, 3), "positive": (4, 5)}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c)).replace(" ", "_").replace("-", "_")


def _match_category(raw: str, categories: tuple[str, ...]) -> str | None:
    """Casa 'Cama, mesa e banho' / 'cama e banho' / 'beleza' com a categoria canônica."""
    key = _fold(raw).replace("_e_", "_")
    for cat in categories:
        if key == cat or key == cat.replace("_e_", "_"):
            return cat
    tokens = [t for t in key.split("_") if len(t) > 2]
    best, best_hits = None, 0
    for cat in categories:
        hits = sum(1 for t in tokens if t in cat.split("_"))
        if hits > best_hits:
            best, best_hits = cat, hits
    return best if best_hits and best_hits >= max(1, len(tokens) - 1) else None


def sanitize(filters: Filters, vocab: Vocabulary) -> tuple[Filters, list[str]]:
    """Remove/corrige valores inválidos. Retorna (filtros válidos, avisos)."""
    f = filters.model_copy()
    notes: list[str] = []

    if f.category:
        cat = _match_category(f.category, vocab.categories)
        if cat != f.category:
            notes.append(f"categoria '{f.category}' -> {cat or 'descartada (não existe na base)'}")
        f.category = cat
    if f.state:
        uf = f.state.strip().upper()
        f.state = uf if uf in vocab.states else None
        if f.state is None:
            notes.append(f"UF '{uf}' descartada")
    if f.year is not None and f.year not in vocab.years:
        notes.append(f"ano {f.year} fora da base ({vocab.years[0]}-{vocab.years[-1]}) descartado")
        f.year = None
    if f.order_status:
        st = f.order_status.strip().lower()
        f.order_status = st if st in vocab.order_statuses else None
        if f.order_status is None:
            notes.append(f"status '{st}' descartado")
    if f.score_min and f.score_max and f.score_min > f.score_max:
        f.score_min, f.score_max = f.score_max, f.score_min
    # Sentimento e faixa de nota redundantes/conflitantes: a nota explícita prevalece.
    if f.sentiment and (f.score_min or f.score_max):
        lo, hi = SCORES_BY_SENTIMENT[f.sentiment]
        if (f.score_min or 1) > hi or (f.score_max or 5) < lo:
            notes.append(f"sentimento '{f.sentiment}' conflita com a faixa de nota; mantida a nota")
            f.sentiment = None
    return f, notes


def merge(base: Filters, overrides: Filters | None) -> Filters:
    """Filtros manuais (interface) sobrescrevem os extraídos pelo LLM."""
    if overrides is None:
        return base
    data = base.model_dump()
    data.update(overrides.active())
    return Filters(**data)


def to_mask(df: pd.DataFrame, f: Filters) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if f.sentiment:
        m &= df["sentiment"] == f.sentiment
    if f.score_min:
        m &= df["review_score"] >= f.score_min
    if f.score_max:
        m &= df["review_score"] <= f.score_max
    if f.year:
        m &= df["review_year"] == f.year
    if f.month:
        m &= df["review_month"] == f.month
    if f.category:
        m &= df["product_categories_pt"].str.split(", ").apply(lambda cats: f.category in cats)
    if f.state:
        m &= df["customer_state"] == f.state
    if f.late_delivery is True:
        m &= df["is_late_delivery"].astype(bool)
    elif f.late_delivery is False:
        # "No prazo" só vale para pedidos entregues — mesma regra do vector store,
        # que omite o atraso de pedidos sem data de entrega.
        m &= ~df["is_late_delivery"].astype(bool) & df["delivery_delay_days"].notna()
    if f.order_status:
        m &= df["order_status"] == f.order_status
    return m


def to_chroma_where(f: Filters) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if f.sentiment:
        clauses.append({"sentiment": {"$eq": f.sentiment}})
    if f.score_min:
        clauses.append({"review_score": {"$gte": f.score_min}})
    if f.score_max:
        clauses.append({"review_score": {"$lte": f.score_max}})
    if f.year:
        clauses.append({"review_year": {"$eq": f.year}})
    if f.month:
        clauses.append({"review_month": {"$eq": f.month}})
    if f.category:
        # O vector store guarda as categorias do pedido numa string ("a, b").
        # Igualdade cobre os ~99% de pedidos com uma única categoria.
        clauses.append({"product_categories_pt": {"$eq": f.category}})
    if f.state:
        clauses.append({"customer_state": {"$eq": f.state}})
    if f.late_delivery is not None:
        clauses.append({"is_late_delivery": {"$eq": f.late_delivery}})
    if f.order_status:
        clauses.append({"order_status": {"$eq": f.order_status}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
