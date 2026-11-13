"""Etapa 3: seleção do que vai pro contexto.

Primeiro o cross-encoder multilíngue re-ranqueia os candidatos: ele lê
pergunta e review juntas, então é bem mais preciso que a similaridade de
vetores do 1º estágio. Depois aplicamos o limiar de relevância
(RERANK_THRESHOLD, calibrado no run_eval), a deduplicação de textos
idênticos depois de normalizados (tem MUITO "ótimo produto" repetido na
base) e o top-N.

Se sobrar menos que MIN_EVIDENCE reviews, a pergunta vira "evidência
insuficiente" (requisito 4 do desafio).
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache

from rag.config import get_settings
from rag.schemas import Candidate


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model, device="cpu")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def rerank(query: str, candidates: list[Candidate], scorer=None) -> list[Candidate]:
    if not candidates:
        return []
    if scorer is None:
        scorer = get_reranker().predict
    scores = scorer([(query, c.text) for c in candidates])
    for c, sc in zip(candidates, scores):
        c.rerank_score = float(sc)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


def select(
    candidates: list[Candidate], threshold: float | None = None, top_n: int | None = None
) -> tuple[list[Candidate], list[Candidate], dict]:
    """Aplica limiar, dedupe e top-N.

    Devolve (selecionados, relevantes_acima_do_limiar, trace). Os
    "relevantes" são todos os que passaram do limiar, sem o corte de top-N:
    são eles que alimentam as estatísticas do contexto.
    """
    s = get_settings()
    threshold = s.rerank_threshold if threshold is None else threshold
    top_n = s.top_n if top_n is None else top_n

    relevant = [c for c in candidates if c.rerank_score is not None and c.rerank_score >= threshold]
    relevant_ids = {id(c) for c in relevant}
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in relevant:
        key = _norm(c.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    selected = deduped[:top_n]

    trace = {
        "limiar": threshold,
        "candidatos": len(candidates),
        "acima_do_limiar": len(relevant),
        "apos_dedupe": len(deduped),
        "selecionados": len(selected),
        "ranking": [
            {"review_id": c.review_id, "rerank": round(c.rerank_score, 3) if c.rerank_score is not None
             and not math.isnan(c.rerank_score) else None, "fontes": c.sources,
             "acima_limiar": id(c) in relevant_ids, "texto": c.text[:120]}
            for c in candidates[:20]
        ],
    }
    return selected, relevant, trace
