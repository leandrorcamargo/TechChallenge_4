"""Fábrica do LLM (Ollama local).

Desligamos o modo thinking do qwen3.8 (reasoning=False): ligado, ele deixa a
resposta bem mais lenta e suja a saída estruturada. Temperatura baixa porque
queremos o modelo colado no contexto, sem criatividade.

Se o Ollama portátil do projeto não estiver rodando, a primeira chamada
sobe ele sozinha.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from rag.config import get_settings


def ensure_ollama_running() -> None:
    from rag.tools import ollama_runtime

    s = get_settings()
    if ollama_runtime.server_version(s) is None:
        ollama_runtime.start_server(s)


@lru_cache(maxsize=4)
def get_chat_model(model: str | None = None, temperature: float | None = None) -> ChatOllama:
    s = get_settings()
    ensure_ollama_running()
    return ChatOllama(
        model=model or s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature if temperature is None else temperature,
        num_ctx=s.llm_num_ctx,
        reasoning=False,
        keep_alive="30m",
    )


def get_analyzer_model() -> ChatOllama:
    s = get_settings()
    return get_chat_model(s.ollama_model_analyzer, temperature=0.0)
