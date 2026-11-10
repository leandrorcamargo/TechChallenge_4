"""Etapa 2 — Recuperação híbrida.

Três listas ranqueadas, fundidas por Reciprocal Rank Fusion (RRF):
  - densa  (HyDE):   embedding da `hypothetical_review` (avaliação hipotética)
  - densa  (query):  embedding da `search_query` reformulada
  - lexical (BM25):  `keywords` + `search_query` sobre o texto das reviews

Por quê híbrida: reviews têm gírias, erros de digitação e expressões literais
("não chegou", "veio quebrado") que o BM25 captura bem, enquanto a busca densa
captura paráfrases. RRF combina rankings sem precisar normalizar scores
(score(d) = Σ 1/(k + rank_r(d)), k=60).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from rag.config import get_settings
from rag.filters import to_mask
from rag.schemas import Candidate, Filters, QueryAnalysis

# Stopwords mínimas em PT — SEM negações ("não", "nunca", "nem"), que são
# decisivas em reclamações (NLP Aula 2: remover stopwords apaga o "não").
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "pra", "pro", "com", "e", "ou", "que", "se", "me", "meu", "minha",
    "eu", "ele", "ela", "foi", "ser", "estar", "esta", "este", "isso", "isto", "ao", "aos", "mas",
    "mais", "muito", "ja", "so", "tambem", "tem", "ter", "sao", "qual", "quais", "como", "sobre",
    "clientes", "cliente", "avaliacoes", "avaliacao", "relatam", "relatados",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25Index:
    """BM25 (Okapi) sobre todo o corpus; filtros aplicados como máscara nos scores."""

    def __init__(self, corpus: pd.DataFrame):
        from rank_bm25 import BM25Okapi

        self.ids = corpus["review_id"].to_numpy()
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in corpus["document_text"]])

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    from rag.data_access import load_corpus

    return BM25Index(load_corpus())


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    """Reciprocal Rank Fusion. Retorna [(id, score, {fonte: rank})] ordenado."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for source, ids in rankings.items():
        for pos, rid in enumerate(ids, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + pos)
            ranks.setdefault(rid, {})[source] = pos
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(rid, sc, ranks[rid]) for rid, sc in ordered]


def hybrid_retrieve(
    analysis: QueryAnalysis, filters: Filters, corpus: pd.DataFrame, dense_fn=None
) -> tuple[list[Candidate], dict]:
    """Executa as 3 buscas e funde. Retorna (candidatos, dados para o trace)."""
    s = get_settings()
    if dense_fn is None:
        from rag.vectorstore_adapter import dense_search as dense_fn

    mask = to_mask(corpus, filters).to_numpy()
    n_filtered = int(mask.sum())

    dense_hyde = dense_fn(analysis.hypothetical_review, s.k_dense, filters, corpus)
    dense_query = dense_fn(analysis.search_query, s.k_dense, filters, corpus)
    bm25_query = " ".join(analysis.keywords) + " " + analysis.search_query
    lexical = get_bm25_index().search(bm25_query, s.k_bm25, mask)

    rankings = {
        "densa_hyde": [rid for rid, _ in dense_hyde],
        "densa_query": [rid for rid, _ in dense_query],
        "bm25": [rid for rid, _ in lexical],
    }
    fused = rrf_fuse(rankings, k=s.rrf_k)[: s.k_fused]

    by_id = corpus.set_index("review_id")
    candidates = []
    for rid, score, ranks in fused:
        if rid not in by_id.index:
            continue
        row = by_id.loc[rid]
        candidates.append(Candidate(review_id=rid, text=row["document_text"], meta=row.to_dict(),
                                    ranks=ranks, rrf_score=score))

    trace = {
        "reviews_que_atendem_filtros": n_filtered,
        "densa_hyde": len(dense_hyde),
        "densa_query": len(dense_query),
        "bm25": len(lexical),
        "consulta_bm25": bm25_query,
        "candidatos_rrf": len(candidates),
        "top_rrf": [
            {"review_id": c.review_id, "rrf": round(c.rrf_score, 4), "fontes": c.sources, "texto": c.text[:120]}
            for c in candidates[:15]
        ],
    }
    return candidates, trace
