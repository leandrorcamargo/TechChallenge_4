"""Bootstrap do ambiente — chamado por scripts/setup.ps1 e scripts/setup.sh
depois que o venv (.venv) foi criado pelo uv.

Etapas (idempotentes — pode rodar de novo sem efeitos colaterais):
  1. Cria .env a partir de .env.example se .env não existir (nunca sobrescreve).
  2. Garante um servidor Ollama: reutiliza um já no ar; senão usa/instala o
     Ollama portátil em .tools/ollama e o inicia em background.
  3. Baixa o(s) modelo(s) LLM (OLLAMA_MODEL e OLLAMA_MODEL_ANALYZER).
  4. Pré-baixa os modelos HuggingFace (reranker e embedding e5) para .tools/hf.
  5. Gera o corpus (Gold local, sem AWS) e o vector store de retrieval/ se faltarem.
  6. Roda o diagnóstico (rag.tools.check_env).
"""

from __future__ import annotations

import argparse
import shutil
import sys

from rag.config import PROJECT_ROOT, get_settings
from rag.tools import check_env, ollama_runtime


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"  {question} -> não (terminal não interativo; use --yes)")
        return False
    return input(f"  {question} [s/N] ").strip().lower() in ("s", "sim", "y", "yes")


def step(title: str) -> None:
    print(f"\n==> {title}", flush=True)


def ensure_env_file() -> None:
    env, example = PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"
    if env.exists():
        print("  .env já existe (mantido).")
    elif example.exists():
        shutil.copyfile(example, env)
        print("  .env criado a partir de .env.example — preencha as credenciais AWS.")


def ensure_ollama(assume_yes: bool) -> bool:
    s = get_settings()
    version = ollama_runtime.server_version(s)
    if version:
        print(f"  Servidor Ollama já no ar (v{version}) em {s.ollama_base_url}.")
        return True

    if ollama_runtime.resolve_binary(s) is None:
        print("  Ollama não encontrado. Será instalado de forma PORTÁTIL em .tools/ollama (~1,6 GB),")
        print("  sem instalar nada no sistema.")
        if not _confirm("Baixar o Ollama portátil agora?", assume_yes):
            print("  Pulado. Rode depois: python -m rag.tools.ollama_runtime install")
            return False
        ollama_runtime.install_portable(s)

    version = ollama_runtime.start_server(s)
    print(f"  Servidor Ollama v{version} no ar em {s.ollama_base_url} (modelos em {s.ollama_models_dir}).")
    return True


def ensure_models(assume_yes: bool) -> None:
    s = get_settings()
    for model in dict.fromkeys([s.ollama_model, s.ollama_model_analyzer]):
        if ollama_runtime.has_model(s, model):
            print(f"  Modelo {model} já disponível.")
            continue
        print(f"  Modelo {model} não encontrado (qwen3.8 = ~18 GB de download).")
        if not _confirm(f"Baixar {model} agora?", assume_yes):
            print(f"  Pulado. Rode depois: python -m rag.tools.ollama_runtime pull {model}")
            continue
        ollama_runtime.pull_model(s, model)


def prefetch_hf_models() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from rag.vectorstore_adapter import retrieval_config

    s = get_settings()
    print(f"  Reranker: {s.reranker_model}")
    CrossEncoder(s.reranker_model)
    embedding = retrieval_config().EMBEDDING_MODEL
    print(f"  Embedding (vector store de retrieval/): {embedding}")
    SentenceTransformer(embedding)


def ensure_data() -> None:
    """Corpus do RAG + vector store da etapa de Embeddings (só constrói o que falta)."""
    from rag import data_access, vectorstore_adapter

    if data_access.corpus_path().exists():
        print(f"  Corpus já existe: {data_access.corpus_path()}")
    else:
        print(f"  Gerando corpus (Gold via '{get_settings().gold_source}')...")
        df = data_access.ensure_corpus()
        print(f"  Corpus: {len(df):,} reviews")

    n = vectorstore_adapter.vector_store_count()
    if n:
        print(f"  Vector store já existe: {n:,} documentos")
    else:
        print("  Construindo o vector store (retrieval/build_vector_store.py, ~5 min em CPU)...")
        vectorstore_adapter.build_vector_store()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepara o ambiente do RAG")
    p.add_argument("--yes", "-y", action="store_true", help="não pergunta; aceita downloads")
    p.add_argument("--skip-model", action="store_true", help="não baixa o LLM (~18 GB)")
    p.add_argument("--skip-ollama", action="store_true", help="não instala/inicia o Ollama")
    p.add_argument("--skip-hf", action="store_true", help="não pré-baixa modelos HuggingFace")
    p.add_argument("--skip-data", action="store_true", help="não gera corpus nem vector store")
    a = p.parse_args(argv)

    step("1/6 Arquivo .env")
    ensure_env_file()

    ollama_ok = False
    if a.skip_ollama:
        step("2/6 Ollama (pulado)")
    else:
        step("2/6 Ollama")
        ollama_ok = ensure_ollama(a.yes)

    if a.skip_model or a.skip_ollama or not ollama_ok:
        step("3/6 Modelo LLM (pulado)")
    else:
        step("3/6 Modelo LLM")
        ensure_models(a.yes)

    if a.skip_hf:
        step("4/6 Modelos HuggingFace (pulado)")
    else:
        step("4/6 Modelos HuggingFace (reranker + embedding)")
        prefetch_hf_models()

    if a.skip_data:
        step("5/6 Corpus e vector store (pulado)")
    else:
        step("5/6 Corpus e vector store")
        ensure_data()

    step("6/6 Diagnóstico")
    smoke = ollama_ok and not a.skip_model
    args = [] if smoke else ["--no-smoke"]
    if a.skip_model:
        args.append("--model-optional")
    return check_env.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
