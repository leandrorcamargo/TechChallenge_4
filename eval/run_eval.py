"""Avaliação do RAG sobre o golden set (eval/golden_questions.yaml).

    python eval/run_eval.py --calibrate     # calibra o RERANK_THRESHOLD (etapas 1-3, sem geração)
    python eval/run_eval.py                 # avaliação completa (pipeline inteiro)
    python eval/run_eval.py --judge         # + LLM-as-judge (fidelidade dos insights)
    python eval/run_eval.py --only pdf-01,oos-01

Mede: acurácia de abstenção (responde quando deve, recusa quando deve —
requisito 4), extração de filtros de metadados, precisão@N lexical do
contexto (proxy: a review selecionada contém termo do tema), taxa de
citações válidas já na saída do LLM e, com --judge, a fidelidade dos
insights. Também mede a latência de cada etapa.

Escreve o relatório em eval/results/report.md e um details.json com a
resposta completa de cada pergunta — quando algo sai errado, é nele que
a gente acha o motivo.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import VocRagPipeline  # noqa: E402
from rag.vectorstore_adapter import retrieval_config  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_questions.yaml"
RESULTS = ROOT / "eval" / "results"


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def load_questions(only: str | None) -> list[dict]:
    qs = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["questions"]
    if only:
        wanted = set(only.split(","))
        qs = [q for q in qs if q["id"] in wanted]
    return qs


# ----------------------------------------------------------------------
# Calibração do limiar de relevância
# ----------------------------------------------------------------------
def calibrate(pipe: VocRagPipeline, questions: list[dict]) -> None:
    """Pra cada pergunta no escopo, mede o score do k-ésimo melhor candidato
    (k = MIN_EVIDENCE). Um limiar bom é um que separa 'answer' de 'sem_evidencia'."""
    from rag import reranking

    k = pipe.settings.min_evidence
    rows = []
    for q in questions:
        if q["group"] in ("fora_do_escopo", "injection"):
            continue
        state = {"question": q["question"], "overrides": None, "trace": []}
        updates, _ = pipe._analyze(state)
        state.update(updates)
        if not state["analysis"].in_scope:
            rows.append((q["id"], q["expected"], None, None, "fora do escopo (análise)"))
            continue
        updates, _ = pipe._retrieve(state)
        ranked = reranking.rerank(state["analysis"].search_query, updates["candidates"],
                                  scorer=pipe.rerank_scorer)
        scores = [c.rerank_score for c in ranked]
        kth = scores[k - 1] if len(scores) >= k else None
        rows.append((q["id"], q["expected"], scores[0] if scores else None, kth,
                     ranked[0].text[:70] if ranked else ""))
        print(f"  {q['id']:<8} {q['expected']:<7} top1={rows[-1][2]!s:<8.6} top{k}={kth!s:<8.6} {rows[-1][4]}",
              flush=True)

    ans = [r[3] for r in rows if r[1] == "answer" and r[3] is not None]
    abst = [r[3] for r in rows if r[1] == "abstain" and r[3] is not None]
    print(f"\nScore do {k}º melhor candidato:")
    if ans:
        print(f"  deve responder: min={min(ans):.3f}  mediana={statistics.median(ans):.3f}")
    if abst:
        print(f"  sem evidência:  max={max(abst):.3f}  mediana={statistics.median(abst):.3f}")
    if ans and abst:
        best, best_acc = None, -1.0
        for t in sorted(set(ans + abst)):
            for cand in (t - 1e-3, t + 1e-3):
                acc = (sum(a >= cand for a in ans) + sum(b < cand for b in abst)) / (len(ans) + len(abst))
                if acc > best_acc:
                    best, best_acc = cand, acc
        print(f"  limiar sugerido: RERANK_THRESHOLD={best:.2f} (acurácia de separação {best_acc:.0%})")


# ----------------------------------------------------------------------
# Avaliação completa
# ----------------------------------------------------------------------
JUDGE_PROMPT = """Você é um avaliador rigoroso. Diga se a AFIRMAÇÃO é sustentada pelas AVALIAÇÕES citadas
e/ou pelas ESTATÍSTICAS (números calculados sobre a base, que o sistema forneceu ao modelo).
Afirmações com inferências, causas ou números que não aparecem nessas fontes NÃO são sustentadas.
Responda apenas "SIM" ou "NAO".

AFIRMAÇÃO: {claim}

AVALIAÇÕES CITADAS:
{reviews}

ESTATÍSTICAS:
{stats}"""


def judge_faithfulness(result, llm) -> tuple[int, int]:
    by_ref = {e.ref: e.text for e in result.evidences}
    stats = next((t.data.get("estatisticas", "") for t in result.trace if t.step.startswith("4.")), "")
    ok = 0
    for ins in result.insights:
        reviews = "\n".join(f"- {by_ref[r]}" for r in ins["citacoes"] if r in by_ref)
        out = llm.invoke(JUDGE_PROMPT.format(claim=ins["texto"], reviews=reviews, stats=stats)).content
        ok += fold(out).strip().startswith("sim")
    return ok, len(result.insights)


def evaluate(pipe: VocRagPipeline, questions: list[dict], judge: bool) -> dict:
    llm = None
    if judge:
        from rag.llm import get_chat_model

        llm = get_chat_model(temperature=0.0)

    details, latencies = [], defaultdict(list)
    for q in questions:
        t0 = time.time()
        r = pipe.run(q["question"])
        abstained = r.status != "answered"
        row = {"id": q["id"], "group": q["group"], "expected": q["expected"], "status": r.status,
               "correct": abstained == (q["expected"] == "abstain"), "seconds": round(time.time() - t0, 1),
               "answer": r.answer, "insights": r.insights}

        steps = {t.step[:2]: t for t in r.trace}
        for t in r.trace:
            latencies[t.step].append(t.duration_ms)
        if "1." in steps and q.get("expected_filters"):
            got = steps["1."].data.get("filtros_finais", {})
            row["filters_expected"] = q["expected_filters"]
            row["filters_got"] = got
            row["filters_ok"] = all(got.get(k) == v for k, v in q["expected_filters"].items())
        if r.status == "answered" and q.get("relevant_terms"):
            terms = [fold(t) for t in q["relevant_terms"]]
            hits = [any(t in fold(e.text) for t in terms) for e in r.evidences]
            row["context_precision_lexical"] = sum(hits) / len(hits) if hits else None
        if "6." in steps:
            v = steps["6."].data
            row["citations_all_valid"] = not v["citacoes_invalidas_removidas"] and \
                not v["insights_descartados_sem_citacao"]
        if judge and r.status == "answered":
            ok, n = judge_faithfulness(r, llm)
            row["faithfulness"] = ok / n if n else None
        details.append(row)
        print(f"  {q['id']:<8} esperado={q['expected']:<7} obtido={r.status:<22} "
              f"{'OK ' if row['correct'] else 'ERRO'} {row['seconds']:>6.1f}s", flush=True)

    return {"details": details, "latency_ms": {k: statistics.mean(v) for k, v in latencies.items()}}


def _mean(values) -> str:
    vals = [v for v in values if v is not None]
    return f"{statistics.mean(vals):.0%} (n={len(vals)})" if vals else "—"


def write_report(res: dict, judge: bool) -> Path:
    s = get_settings()
    d = res["details"]
    by_group = defaultdict(list)
    for row in d:
        by_group[row["group"]].append(row["correct"])

    lines = [
        "# Avaliação do RAG — Voice of Customer",
        "",
        f"- Data: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- LLM: `{s.ollama_model}` · reranker: `{s.reranker_model}` · embedding: "
        f"`{retrieval_config().EMBEDDING_MODEL}` (vector store de retrieval/)",
        f"- RERANK_THRESHOLD={s.rerank_threshold} · MIN_EVIDENCE={s.min_evidence} · TOP_N={s.top_n}",
        "",
        "## Métricas",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Acurácia de abstenção (geral) | {_mean([r['correct'] for r in d])} |",
        f"| Extração de filtros de metadados | {_mean([r.get('filters_ok') for r in d])} |",
        f"| Precisão@N lexical do contexto | {_mean([r.get('context_precision_lexical') for r in d])} |",
        f"| Respostas com 100% das citações válidas (antes da pós-validação) | "
        f"{_mean([r.get('citations_all_valid') for r in d])} |",
    ]
    if judge:
        lines.append(f"| Fidelidade dos insights (LLM-as-judge) | {_mean([r.get('faithfulness') for r in d])} |")
    lines += ["", "### Acurácia por grupo", "", "| Grupo | Acurácia |", "|---|---|"]
    lines += [f"| {g} | {_mean(v)} |" for g, v in by_group.items()]
    lines += ["", "### Latência média por etapa", "", "| Etapa | ms |", "|---|---|"]
    lines += [f"| {k} | {v:,.0f} |" for k, v in res["latency_ms"].items()]
    lines += ["", "## Perguntas", "", "| id | esperado | obtido | ok | filtros | precisão ctx | s |",
              "|---|---|---|---|---|---|---|"]
    for r in d:
        prec = r.get("context_precision_lexical")
        lines.append(f"| {r['id']} | {r['expected']} | {r['status']} | {'✅' if r['correct'] else '❌'} | "
                     f"{'' if 'filters_ok' not in r else ('✅' if r['filters_ok'] else '❌ ' + str(r['filters_got']))}"
                     f" | {'' if prec is None else f'{prec:.0%}'} | {r['seconds']} |")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = RESULTS / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "details.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--only", default=None, help="ids separados por vírgula")
    a = p.parse_args()

    questions = load_questions(a.only)
    pipe = VocRagPipeline()
    if a.calibrate:
        calibrate(pipe, questions)
        return 0
    res = evaluate(pipe, questions, a.judge)
    print(f"\nRelatório: {write_report(res, a.judge)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
