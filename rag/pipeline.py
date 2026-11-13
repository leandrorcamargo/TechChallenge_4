"""Pipeline RAG completo, composto em LCEL (LangChain Expression Language).

    pergunta
      -> 0. guardrail de entrada (prompt injection)        --bloqueada--> recusa
      -> 1. processamento da pergunta (LLM, QueryAnalysis)  --fora de escopo--> recusa
      -> 2. recuperação híbrida (densa HyDE + densa query + BM25, RRF)
      -> 3. seleção (cross-encoder rerank, limiar, dedupe, top-N)
                                                           --evidência insuficiente--> recusa
      -> 4. construção do contexto (XML [R#] + estatísticas)
      -> 5. geração (LLM, VocAnswer)
      -> 6. validação de citações                          --sem citação válida--> recusa
      -> RagResult (resposta + insights + evidências + trace de cada etapa)

Cada etapa é um RunnableLambda que recebe e devolve o estado (dict) e vai
registrando tempo e dados no trace — é esse trace que a interface mostra
no "Inspetor do pipeline" (e que salvou a vida na hora de debugar).

CLI:
    python -m rag.pipeline "Quais reclamações relacionadas à entrega aparecem com maior frequência?"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any, Callable

import pandas as pd
from langchain_core.runnables import RunnableBranch, RunnableLambda

from rag import context_builder, guardrails, query_processing, reranking
from rag.config import get_settings
from rag.filters import merge
from rag.generation import build_generation_chain, render_prompt
from rag.prompts import (MSG_INJECTION, MSG_INSUFFICIENT, MSG_NO_VALID_CITATIONS, MSG_OUT_OF_SCOPE,
                         PROMPT_VERSION)
from rag.retrieval import hybrid_retrieve
from rag.schemas import Filters, RagResult, StepTrace

State = dict[str, Any]


def _step(name: str, fn: Callable[[State], dict]) -> RunnableLambda:
    def run(state: State) -> State:
        t0 = time.perf_counter()
        updates, trace_data = fn(state)
        ms = (time.perf_counter() - t0) * 1000
        state = {**state, **updates}
        state["trace"] = [*state["trace"], StepTrace(name, round(ms, 1), trace_data)]
        return state

    return RunnableLambda(run, name=name)


class VocRagPipeline:
    """Monta a chain. As dependências entram por parâmetro porque os testes
    passam versões falsas (LLM falso, busca falsa, reranker falso)."""

    def __init__(self, corpus: pd.DataFrame | None = None, analyzer_chain=None, generation_chain=None,
                 dense_fn=None, rerank_scorer=None):
        self.settings = get_settings()
        if corpus is None:
            from rag.data_access import load_corpus

            corpus = load_corpus()
        self.corpus = corpus
        self._analyzer_chain = analyzer_chain
        self._generation_chain = generation_chain
        self.dense_fn = dense_fn
        self.rerank_scorer = rerank_scorer
        self.chain = self._build()

    # ---- preguiçoso de propósito: o LLM só carrega se for usado ----
    @property
    def analyzer_chain(self):
        if self._analyzer_chain is None:
            self._analyzer_chain = query_processing.build_analyzer_chain()
        return self._analyzer_chain

    @property
    def generation_chain(self):
        if self._generation_chain is None:
            self._generation_chain = build_generation_chain()
        return self._generation_chain

    # ---- etapas ----
    def _guard(self, s: State):
        blocked = guardrails.detect_injection(s["question"])
        return {"blocked": blocked}, {"prompt_injection_detectado": blocked}

    def _analyze(self, s: State):
        analysis, filters, trace = query_processing.analyze(s["question"], self.analyzer_chain)
        if s.get("overrides"):
            filters = merge(filters, s["overrides"])
            trace["filtros_manuais_interface"] = s["overrides"].active()
            trace["filtros_finais"] = filters.active()
        return {"analysis": analysis, "filters": filters}, trace

    def _retrieve(self, s: State):
        candidates, trace = hybrid_retrieve(s["analysis"], s["filters"], self.corpus, dense_fn=self.dense_fn)
        return {"candidates": candidates}, trace

    def _select(self, s: State):
        # comparamos com a pergunta reformulada, que é autocontida (sem ruído
        # de conversa), não com a pergunta crua do usuário
        ranked = reranking.rerank(s["analysis"].search_query, s["candidates"], scorer=self.rerank_scorer)
        selected, relevant, trace = reranking.select(ranked)
        trace["min_evidencias"] = self.settings.min_evidence
        trace["consulta_rerank"] = s["analysis"].search_query
        return {"ranked": ranked, "selected": selected, "relevant": relevant}, trace

    def _context(self, s: State):
        evidences = context_builder.build_evidences(s["selected"])
        reviews = context_builder.format_reviews(evidences)
        stats = context_builder.build_stats(self.corpus, s["filters"], s["relevant"], len(evidences))
        prompt_text = render_prompt(s["question"], reviews, stats)
        trace = {"n_reviews_no_contexto": len(evidences), "estatisticas": stats,
                 "mapa_ids": {e.ref: e.review_id for e in evidences}, "prompt_versao": PROMPT_VERSION,
                 "prompt_enviado_ao_llm": prompt_text, "tamanho_prompt_chars": len(prompt_text)}
        return {"evidences": evidences, "reviews_xml": reviews, "stats": stats}, trace

    def _generate(self, s: State):
        answer = self.generation_chain.invoke(
            {"question": s["question"], "reviews": s["reviews_xml"], "stats": s["stats"]}
        )
        return {"answer": answer}, {"modelo": self.settings.ollama_model, "resposta_bruta": answer.model_dump()}

    def _validate(self, s: State):
        valid_refs = {e.ref for e in s["evidences"]}
        insights, report = guardrails.validate_citations(s["answer"], valid_refs)
        cited = {r for ins in insights for r in ins["citacoes"]}
        for e in s["evidences"]:
            e.cited = e.ref in cited
        report["evidencia_suficiente_segundo_llm"] = s["answer"].evidencia_suficiente
        return {"insights": insights}, report

    # ---- finais ----
    @staticmethod
    def _finish_injection(s: State) -> RagResult:
        return RagResult(s["question"], "out_of_scope", MSG_INJECTION, trace=s["trace"])

    @staticmethod
    def _finish_out_of_scope(s: State) -> RagResult:
        reason = s["analysis"].out_of_scope_reason
        msg = MSG_OUT_OF_SCOPE.format(reason=f" Motivo: {reason}." if reason else "")
        return RagResult(s["question"], "out_of_scope", msg, trace=s["trace"])

    def _finish_insufficient(self, s: State) -> RagResult:
        # mostra o que achou de mais próximo, sem tirar conclusão de nada
        near = context_builder.build_evidences(s["ranked"][:5])
        msg = MSG_INSUFFICIENT.format(n=len(s["selected"]), min_n=self.settings.min_evidence)
        return RagResult(s["question"], "insufficient_evidence", msg, evidences=near, trace=s["trace"])

    @staticmethod
    def _finish_answered(s: State) -> RagResult:
        answer = s["answer"]
        if not answer.evidencia_suficiente:
            return RagResult(s["question"], "insufficient_evidence", answer.resposta,
                             evidences=s["evidences"], trace=s["trace"])
        if not s["insights"]:
            return RagResult(s["question"], "insufficient_evidence", MSG_NO_VALID_CITATIONS,
                             evidences=s["evidences"], trace=s["trace"])
        return RagResult(s["question"], "answered", answer.resposta, insights=s["insights"],
                         confidence=answer.nivel_confianca, evidences=s["evidences"], trace=s["trace"])

    def _build(self):
        min_ev = self.settings.min_evidence
        answer_branch = (
            _step("4. Construção do contexto", self._context)
            | _step("5. Geração (LLM)", self._generate)
            | _step("6. Validação das citações", self._validate)
            | RunnableLambda(self._finish_answered)
        )
        retrieval_branch = (
            _step("2. Recuperação híbrida", self._retrieve)
            | _step("3. Seleção (rerank)", self._select)
            | RunnableBranch((lambda s: len(s["selected"]) < min_ev, RunnableLambda(self._finish_insufficient)),
                             answer_branch)
        )
        analysis_branch = (
            _step("1. Processamento da pergunta", self._analyze)
            | RunnableBranch((lambda s: not s["analysis"].in_scope, RunnableLambda(self._finish_out_of_scope)),
                             retrieval_branch)
        )
        return (
            _step("0. Guardrail de entrada", self._guard)
            | RunnableBranch((lambda s: s["blocked"], RunnableLambda(self._finish_injection)), analysis_branch)
        )

    def run(self, question: str, overrides: Filters | None = None) -> RagResult:
        question = question.strip()
        return self.chain.invoke({"question": question, "overrides": overrides, "trace": []})


def _print(result: RagResult) -> None:
    print(f"\n[{result.status}] confiança={result.confidence}  ({result.total_ms / 1000:.1f}s)\n")
    print(result.answer, "\n")
    for ins in result.insights:
        print(f"  - {ins['texto']}  [{', '.join(ins['citacoes'])}]")
    if result.evidences:
        print("\nEvidências:")
        for e in result.evidences:
            mark = "*" if e.cited else " "
            print(f" {mark}{e.ref:<4} {e.review_score}★ {e.category:<24} {e.review_id}  {e.text[:90]!r}")
    print("\nTempo por etapa:")
    for t in result.trace:
        print(f"  {t.step:<32} {t.duration_ms:>9.0f} ms")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pergunta ao RAG de Voice of Customer")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="imprime o RagResult completo em JSON")
    a = p.parse_args(argv)
    result = VocRagPipeline().run(a.question)
    if a.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    else:
        _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
