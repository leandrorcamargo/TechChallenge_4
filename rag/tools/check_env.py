"""Diagnóstico do ambiente: imprime uma tabela OK / AVISO / FALHOU.

    python -m rag.tools.check_env            # inclui smoke test do LLM
    python -m rag.tools.check_env --no-smoke

Código de saída 1 se algum item obrigatório FALHOU.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time

from rag.config import PROJECT_ROOT, get_settings

OK, WARN, FAIL = "OK", "AVISO", "FALHOU"
REQUIRED_PACKAGES = [
    "langchain_core", "langchain_ollama", "chromadb", "sentence_transformers",
    "torch", "rank_bm25", "pandas", "pyarrow", "boto3", "pydantic", "streamlit", "yaml",
]
REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"]


def _version(mod) -> str:
    return str(getattr(mod, "__version__", "")) or "instalado"


def run_checks(smoke: bool, model_optional: bool = False) -> list[tuple[str, str, str]]:
    import os

    s = get_settings()
    rows: list[tuple[str, str, str]] = []

    v = sys.version_info
    rows.append(("Python", OK if (3, 11) <= v[:2] < (3, 13) else FAIL, sys.version.split()[0]))
    rows.append(("venv", OK if sys.prefix != sys.base_prefix else WARN, sys.prefix))

    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            missing.append(name)
    rows.append(("Dependências Python", FAIL if missing else OK,
                 "faltando: " + ", ".join(missing) if missing else f"{len(REQUIRED_PACKAGES)} pacotes"))

    env_file = PROJECT_ROOT / ".env"
    if s.gold_source == "s3":
        absent = [k for k in REQUIRED_ENV if not os.environ.get(k)]
        rows.append((".env (AWS)", FAIL if absent else OK,
                     "GOLD_SOURCE=s3 exige: " + ", ".join(absent) if absent else "credenciais presentes"))
    else:
        rows.append((".env", OK if env_file.exists() else WARN,
                     "Gold local (GOLD_SOURCE=local): credenciais AWS não são necessárias"))

    from rag.data_access import corpus_path
    from rag.vectorstore_adapter import retrieval_config, vector_store_count

    cp = corpus_path()
    rows.append(("Corpus do RAG", OK if cp.exists() else FAIL,
                 str(cp.relative_to(PROJECT_ROOT)) if cp.exists() else "ausente: python -m rag.data_access"))
    n_docs = vector_store_count()
    rows.append(("Vector store (retrieval/)", OK if n_docs else FAIL,
                 f"{n_docs:,} documentos · {retrieval_config().EMBEDDING_MODEL}" if n_docs
                 else "ausente: python -m rag.vectorstore_adapter build"))

    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        rows.append(("GPU", OK, out.stdout.strip() or "NVIDIA detectada"))
    else:
        rows.append(("GPU", WARN, "sem NVIDIA — o LLM roda em CPU (mais lento)"))

    from rag.tools import ollama_runtime

    binary = ollama_runtime.resolve_binary(s)
    version = ollama_runtime.server_version(s)
    started = ""
    if version is None and binary is not None:
        # Mesmo comportamento da interface/pipeline: o servidor sobe sob demanda.
        try:
            version = ollama_runtime.start_server(s)
            started = " (iniciado agora)"
        except RuntimeError as e:
            started = f" — falhou ao iniciar: {e}"
    rows.append(("Ollama (binário)", OK if binary or version else FAIL, str(binary or "não encontrado")))
    rows.append(("Ollama (servidor)", OK if version else FAIL,
                 f"v{version} em {s.ollama_base_url}{started}" if version
                 else f"parado em {s.ollama_base_url}{started}"))

    model_ok = False
    if version:
        for model in dict.fromkeys([s.ollama_model, s.ollama_model_analyzer]):
            model_ok = ollama_runtime.has_model(s, model)
            missing_status = WARN if model_optional else FAIL
            rows.append((f"Modelo {model}", OK if model_ok else missing_status,
                         "disponível" if model_ok else f"python -m rag.tools.ollama_runtime pull {model}"))

    if smoke and version and model_ok:
        rows.append(smoke_test())
    elif smoke:
        rows.append(("Smoke test LLM", FAIL, "pulado: servidor ou modelo indisponível"))
    return rows


def smoke_test() -> tuple[str, str, str]:
    """Geração com structured output e thinking desligado — o mesmo modo usado no RAG."""
    from pydantic import BaseModel, Field

    from rag.llm import get_chat_model

    class Ping(BaseModel):
        resposta: str = Field(description="a palavra 'ok'")
        numero: int = Field(description="o resultado de 2+3")

    try:
        llm = get_chat_model().with_structured_output(Ping)
        t0 = time.time()
        out = llm.invoke("Responda com resposta='ok' e numero=2+3.")
        dt = time.time() - t0
        ok = isinstance(out, Ping) and out.numero == 5
        return ("Smoke test LLM", OK if ok else FAIL, f"{out!r} em {dt:.1f}s")
    except Exception as e:  # noqa: BLE001
        return ("Smoke test LLM", FAIL, f"{type(e).__name__}: {e}"[:200])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-smoke", action="store_true", help="não executa o smoke test do LLM")
    p.add_argument("--model-optional", action="store_true", help="modelo ausente vira AVISO (setup --skip-model)")
    a = p.parse_args(argv)

    rows = run_checks(smoke=not a.no_smoke, model_optional=a.model_optional)
    width = max(len(r[0]) for r in rows)
    print()
    for name, status, detail in rows:
        print(f"  [{status:^6}] {name:<{width}}  {detail}")
    failed = [r for r in rows if r[1] == FAIL]
    print(f"\n  {'TUDO PRONTO' if not failed else f'{len(failed)} item(ns) com FALHA'}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
