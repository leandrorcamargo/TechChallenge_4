"""Etapa 5 — Geração da resposta com o LLM (Ollama / qwen3.8).

Saída estruturada (VocAnswer, Pydantic) via JSON schema do Ollama: resposta,
insights com citações [R#], nível de confiança e a auto-declaração de
evidência suficiente.
"""

from __future__ import annotations

from rag.prompts import GENERATION_PROMPT
from rag.schemas import VocAnswer


def build_generation_chain(llm=None):
    if llm is None:
        from rag.llm import get_chat_model

        llm = get_chat_model()
    return GENERATION_PROMPT | llm.with_structured_output(VocAnswer).with_retry(stop_after_attempt=2)


def render_prompt(question: str, reviews: str, stats: str) -> str:
    """Texto exato enviado ao modelo (para o inspetor do pipeline)."""
    msgs = GENERATION_PROMPT.format_messages(question=question, reviews=reviews, stats=stats)
    return "\n\n".join(f"[{m.type.upper()}]\n{m.content}" for m in msgs)
