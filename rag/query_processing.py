"""Etapa 1 — Processamento e transformação da pergunta.

O LLM (structured output -> QueryAnalysis) faz, numa única chamada:
  - classificação de escopo (guardrail de entrada semântico);
  - classificação de intenção;
  - reescrita da consulta (search_query), HyDE (hypothetical_review) e
    expansão lexical (keywords) — uma consulta por estratégia de busca;
  - extração de filtros de metadados (self-query).

Em código, os filtros são validados contra o vocabulário real da Gold (o LLM
não consegue inventar categoria/UF/ano inexistente) e aplicam-se defaults de
intenção. Se o LLM falhar, cai para uma análise mínima (a pergunta original).
"""

from __future__ import annotations

import re

from rag.data_access import load_vocabulary
from rag.filters import sanitize
from rag.prompts import ANALYZER_PROMPT
from rag.retrieval import tokenize
from rag.schemas import Filters, QueryAnalysis


def build_analyzer_chain(llm=None):
    if llm is None:
        from rag.llm import get_analyzer_model

        llm = get_analyzer_model()
    vocab = load_vocabulary()
    prompt = ANALYZER_PROMPT.partial(categories=", ".join(vocab.categories))
    return prompt | llm.with_structured_output(QueryAnalysis).with_retry(stop_after_attempt=2)


def fallback_analysis(question: str) -> QueryAnalysis:
    return QueryAnalysis(in_scope=True, intent="outro", search_query=question,
                         hypothetical_review=question, keywords=tokenize(question)[:12], filters=Filters())


def apply_intent_defaults(analysis: QueryAnalysis) -> tuple[Filters, list[str]]:
    """Sem filtro de nota/sentimento explícito: 'problemas' foca notas <=3, 'elogios' foca positivas."""
    f = analysis.filters.model_copy()
    notes = []
    if f.sentiment is None and f.score_min is None and f.score_max is None:
        if analysis.intent == "problemas":
            f.score_max = 3
            notes.append("intenção 'problemas' -> nota <= 3 (default)")
        elif analysis.intent == "elogios":
            f.sentiment = "positive"
            notes.append("intenção 'elogios' -> sentimento positivo (default)")
    return f, notes


_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_UF_RE = re.compile(r"\b([A-Z]{2})\b")


def apply_explicit_mentions(question: str, filters: Filters, vocab) -> tuple[Filters, list[str]]:
    """Complementa (sem sobrescrever) o LLM com menções inequívocas na pergunta:
    um único ano da base (ex.: 'em 2017') e uma única sigla de UF em maiúsculas (ex.: 'SP')."""
    f = filters.model_copy()
    notes = []
    years = {int(y) for y in _YEAR_RE.findall(question)} & set(vocab.years)
    if f.year is None and len(years) == 1:
        f.year = years.pop()
        notes.append(f"ano {f.year} citado na pergunta -> filtro (regra determinística)")
    ufs = set(_UF_RE.findall(question)) & set(vocab.states)
    if f.state is None and len(ufs) == 1:
        f.state = ufs.pop()
        notes.append(f"UF {f.state} citada na pergunta -> filtro (regra determinística)")
    return f, notes


def analyze(question: str, chain=None) -> tuple[QueryAnalysis, Filters, dict]:
    """Retorna (análise, filtros finais validados, dados para o trace)."""
    error = None
    try:
        analysis = (chain or build_analyzer_chain()).invoke({"question": question})
    except Exception as e:  # noqa: BLE001 — degrada para busca com a pergunta original
        analysis, error = fallback_analysis(question), f"{type(e).__name__}: {e}"

    vocab = load_vocabulary()
    filters, notes = apply_intent_defaults(analysis)
    filters, explicit_notes = apply_explicit_mentions(question, filters, vocab)
    filters, sanitize_notes = sanitize(filters, vocab)
    notes += explicit_notes
    trace = {
        "pergunta_original": question,
        "in_scope": analysis.in_scope,
        "motivo_fora_escopo": analysis.out_of_scope_reason,
        "intencao": analysis.intent,
        "search_query": analysis.search_query,
        "hypothetical_review (HyDE)": analysis.hypothetical_review,
        "keywords": analysis.keywords,
        "filtros_extraidos_llm": analysis.filters.active(),
        "filtros_finais": filters.active(),
        "ajustes": notes + sanitize_notes,
    }
    if error:
        trace["erro_llm_fallback"] = error
    return analysis, filters, trace
