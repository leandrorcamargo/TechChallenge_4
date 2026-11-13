"""Etapa 1: processamento e transformação da pergunta.

Numa única chamada o LLM (structured output -> QueryAnalysis) já devolve:
escopo da pergunta, intenção, a pergunta reescrita pra busca (search_query),
uma review hipotética (HyDE), palavras-chave e os filtros de metadados
(self-query).

Em código a gente desconfia do LLM: os filtros são validados contra o
vocabulário real da Gold (categoria/UF/ano inventado não passa) e a
intenção aplica defaults. Se a chamada falhar, cai pra uma análise mínima
com a pergunta original — melhor responder mal do que não responder.
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
    """O LLM às vezes esquece do ano/UF que está escrito na própria pergunta;
    aqui a gente só completa, sem sobrescrever: um único ano da base e uma
    única sigla de UF em maiúsculas (duas menções = ambíguo, não mexe)."""
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
