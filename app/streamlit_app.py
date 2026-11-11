"""Interface de demonstração — Voice of Customer Intelligence com RAG.

    streamlit run app/streamlit_app.py

- Chat com respostas fundamentadas e citações [R#].
- Painel de evidências: as avaliações usadas, com review_id e metadados.
- Inspetor do pipeline: o que aconteceu em cada etapa (pergunta transformada,
  filtros, candidatos de cada busca, rerank e limiar, prompt exato enviado ao
  LLM, resposta bruta, validação de citações e latência).
- Filtros manuais de metadados (sobrescrevem os extraídos da pergunta).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = (Path(__file__).resolve().parent.parent if "__file__" in globals()
            else Path("/Workspace/Users/fvieirasanches@gmail.com/TechChallenge_4"))
sys.path.insert(0, str(_ROOT))

from rag.config import get_settings  # noqa: E402
from rag.schemas import Filters, RagResult  # noqa: E402

st.set_page_config(page_title="VoC Intelligence — Olist", page_icon="💬", layout="wide")

EXAMPLES = {
    "Investigação de reclamações": "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "Tema específico": "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Experiências positivas": "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Insatisfação": "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Evidências": "Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
                  "relacionados ao prazo de entrega?",
    "Metadados: categoria": "O que os clientes mais elogiam em produtos de cama, mesa e banho?",
    "Metadados: nota baixa": "Quais são os principais problemas relatados nas avaliações com nota baixa?",
    "Fora do escopo": "Qual é a cotação do dólar hoje?",
}
STATUS_LABEL = {
    "answered": ("Resposta fundamentada", "green"),
    "out_of_scope": ("Fora do escopo", "orange"),
    "insufficient_evidence": ("Evidência insuficiente", "red"),
}


@st.cache_resource(show_spinner="Carregando corpus, índices e modelos (primeira vez demora)...")
def load_pipeline():
    from rag.pipeline import VocRagPipeline
    from rag.reranking import get_reranker
    from rag.retrieval import get_bm25_index
    from rag.vectorstore_adapter import warmup

    get_bm25_index()
    get_reranker()
    warmup()  # carrega o embedding e5 e abre o vector store de retrieval/
    return VocRagPipeline()


@st.cache_data
def vocabulary():
    from rag.data_access import load_vocabulary

    v = load_vocabulary()
    return list(v.categories), list(v.states), list(v.years)


def sidebar_filters() -> Filters | None:
    st.sidebar.header("Filtros manuais")
    st.sidebar.caption("Opcional: sobrescrevem os filtros extraídos automaticamente da pergunta.")
    cats, states, years = vocabulary()
    sentiment = st.sidebar.selectbox("Sentimento", ["(auto)", "negative", "neutral", "positive"])
    use_score = st.sidebar.checkbox("Restringir nota")
    score = st.sidebar.slider("Nota", 1, 5, (1, 5), disabled=not use_score)
    category = st.sidebar.selectbox("Categoria", ["(auto)", *cats])
    state = st.sidebar.selectbox("UF do cliente", ["(auto)", *states])
    year = st.sidebar.selectbox("Ano da avaliação", ["(auto)", *years])
    late = st.sidebar.selectbox("Entrega atrasada", ["(auto)", "sim", "não"])

    f = Filters(
        sentiment=None if sentiment == "(auto)" else sentiment,
        score_min=score[0] if use_score else None,
        score_max=score[1] if use_score else None,
        category=None if category == "(auto)" else category,
        state=None if state == "(auto)" else state,
        year=None if year == "(auto)" else int(year),
        late_delivery=None if late == "(auto)" else late == "sim",
    )
    return f if f.active() else None


def render_result(r: RagResult) -> None:
    label, color = STATUS_LABEL[r.status]
    meta = f":{color}-badge[{label}]"
    if r.confidence:
        meta += f" &nbsp; Confiança: **{r.confidence}**"
    meta += f" &nbsp; ⏱ {r.total_ms / 1000:.1f}s"
    st.markdown(meta)
    st.markdown(r.answer)

    if r.insights:
        st.markdown("**Insights e evidências**")
        for ins in r.insights:
            refs = " ".join(f"`{c}`" for c in ins["citacoes"])
            st.markdown(f"- {ins['texto']} {refs}")

    tabs = st.tabs(["📄 Evidências", "🔍 Inspetor do pipeline", "⏱ Latência"])
    with tabs[0]:
        if not r.evidences:
            st.info("Nenhuma avaliação foi recuperada para esta pergunta.")
        else:
            if r.status != "answered":
                st.caption("Avaliações mais próximas encontradas — insuficientes para sustentar uma conclusão.")
            df = pd.DataFrame([{
                "ref": e.ref, "citada": "✅" if e.cited else "", "nota": e.review_score,
                "avaliação": e.text, "categoria": e.category, "data": e.date, "UF": e.state,
                "atrasou": "sim" if e.late_delivery else "não",
                "relevância (rerank)": None if e.rerank_score is None else round(e.rerank_score, 2),
                "recuperada por": e.sources, "review_id": e.review_id,
            } for e in r.evidences])
            st.dataframe(df, hide_index=True, width="stretch",
                         column_config={"avaliação": st.column_config.TextColumn(width="large")})
    with tabs[1]:
        for t in r.trace:
            with st.expander(f"{t.step} — {t.duration_ms:,.0f} ms"):
                data = dict(t.data)
                prompt = data.pop("prompt_enviado_ao_llm", None)
                stats = data.pop("estatisticas", None)
                for key in ("top_rrf", "ranking"):
                    if key in data:
                        st.markdown(f"**{key}**")
                        st.dataframe(pd.DataFrame(data.pop(key)), hide_index=True, width="stretch")
                st.json(json.loads(json.dumps(data, default=str, ensure_ascii=False)), expanded=True)
                if stats:
                    st.markdown("**Estatísticas enviadas ao LLM**")
                    st.code(stats, language="markdown")
                if prompt:
                    st.markdown("**Prompt exato enviado ao LLM**")
                    st.code(prompt, language="xml")
    with tabs[2]:
        lat = pd.DataFrame({"etapa": [t.step for t in r.trace], "ms": [t.duration_ms for t in r.trace]})
        st.bar_chart(lat, x="etapa", y="ms", horizontal=True)


def main() -> None:
    s = get_settings()
    st.title("💬 Voice of Customer Intelligence")
    st.caption(f"RAG sobre 42 mil avaliações de clientes da Olist (2016–2018) · LLM local: `{s.ollama_model}` "
               f"via Ollama · respostas sempre com evidências rastreáveis")

    overrides = sidebar_filters()
    st.sidebar.header("Perguntas de exemplo")
    clicked = None
    for label, q in EXAMPLES.items():
        if st.sidebar.button(label, width="stretch"):
            clicked = q
    if st.sidebar.button("Limpar conversa", type="secondary"):
        st.session_state.history = []

    history: list[RagResult] = st.session_state.setdefault("history", [])
    for r in history:
        with st.chat_message("user"):
            st.markdown(r.question)
        with st.chat_message("assistant"):
            render_result(r)

    question = st.chat_input("Pergunte sobre as avaliações dos clientes...") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                pipeline = load_pipeline()
                with st.spinner("Analisando pergunta, recuperando avaliações e gerando resposta..."):
                    result = pipeline.run(question, overrides=overrides)
            except Exception as e:  # noqa: BLE001
                st.error(f"Erro ao executar o pipeline: {type(e).__name__}: {e}")
                if "are available" in str(e) and "GiB" in str(e):
                    st.caption(
                        "O Ollama não tem memória suficiente para carregar este modelo. "
                        "Opções: (1) `ollama ps` + `ollama stop <modelo>` para descarregar modelos "
                        "e fechar apps pesados; (2) trocar para um modelo menor adicionando "
                        "`OLLAMA_MODEL=qwen3:4b` no `.env` da raiz (após `ollama pull qwen3:4b`); "
                        "(3) reduzir o contexto com `LLM_NUM_CTX=4096` no `.env`. "
                        "Reinicie o app após mudar o `.env`."
                    )
                else:
                    st.caption("Verifique o ambiente com `python -m rag.tools.check_env`.")
                return
            render_result(result)
        history.append(result)


main()
