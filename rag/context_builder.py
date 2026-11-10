"""Etapa 4 — Construção do contexto enviado ao LLM.

- Cada review selecionada recebe um ID curto (R1..Rn) mapeado para o review_id
  real: menos tokens e menos erros de citação, mantendo rastreabilidade total.
- Metadados relevantes vão como atributos XML (nota, categoria, data, UF, atraso).
- Bloco <estatisticas> calculado em pandas: top-k não mede frequência, então o
  LLM recebe contagens reais em vez de inventá-las.
"""

from __future__ import annotations

from collections import Counter
from html import escape

import pandas as pd

from rag.filters import to_mask
from rag.schemas import Candidate, Evidence, Filters


def _evidence(ref: str, c: Candidate) -> Evidence:
    m = c.meta
    return Evidence(
        ref=ref, review_id=c.review_id, text=c.text, review_score=int(m["review_score"]),
        category=str(m.get("product_category_main", "")), date=str(m.get("review_creation_date", "")),
        state=str(m.get("customer_state", "")), late_delivery=bool(m.get("is_late_delivery", False)),
        rerank_score=c.rerank_score, sources=c.sources,
    )


def build_evidences(selected: list[Candidate]) -> list[Evidence]:
    return [_evidence(f"R{i}", c) for i, c in enumerate(selected, start=1)]


def format_reviews(evidences: list[Evidence]) -> str:
    lines = []
    for e in evidences:
        attrs = (f'id="{e.ref}" nota="{e.review_score}" categoria="{escape(e.category)}" '
                 f'data="{e.date}" uf="{e.state}" entrega_atrasada="{"sim" if e.late_delivery else "não"}"')
        lines.append(f"<review {attrs}>{escape(e.text, quote=False)}</review>")
    return "\n".join(lines)


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total:.0%})" if total else "0"


def build_stats(corpus: pd.DataFrame, filters: Filters, relevant: list[Candidate], selected_n: int) -> str:
    n_total = len(corpus)
    n_filtered = int(to_mask(corpus, filters).sum())
    active = filters.active()
    lines = [
        f"- Avaliações com texto na base: {n_total:,}".replace(",", "."),
        f"- Filtros aplicados: {active if active else 'nenhum'}",
        f"- Avaliações que atendem aos filtros: {n_filtered:,}".replace(",", "."),
        f"- Avaliações relevantes para a pergunta (acima do limiar de relevância, entre as "
        f"candidatas recuperadas): {len(relevant)}; enviadas neste contexto: {selected_n}",
    ]
    if relevant:
        k = len(relevant)
        notes = Counter(int(c.meta["review_score"]) for c in relevant)
        late = sum(bool(c.meta.get("is_late_delivery")) for c in relevant)
        cats = Counter(str(c.meta.get("product_category_main")) for c in relevant).most_common(5)
        lines += [
            "- Entre as avaliações relevantes:",
            "  - notas: " + ", ".join(f"{n}★={notes.get(n, 0)}" for n in range(1, 6)),
            f"  - pedidos entregues com atraso: {_pct(late, k)}",
            "  - categorias mais frequentes: " + ", ".join(f"{c} ({n})" for c, n in cats),
        ]
    lines.append("- Observação: as contagens de 'relevantes' referem-se às avaliações recuperadas para "
                 "esta pergunta, não ao total da base.")
    return "\n".join(lines)
