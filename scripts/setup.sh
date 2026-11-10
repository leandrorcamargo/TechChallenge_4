#!/usr/bin/env bash
# Prepara TODO o ambiente do projeto no Linux/macOS, de forma isolada (estilo venv).
#
# Nada é instalado no sistema. Tudo fica dentro do projeto:
#   .tools/uv             uv (gerenciador de Python/venv), versão fixada
#   .tools/python         Python 3.11 gerenciado pelo uv
#   .venv                 ambiente virtual com dependências travadas (uv.lock)
#   .tools/ollama         Ollama portátil (binário oficial, SHA-256 verificado)
#   .tools/ollama/models  modelos LLM (qwen3.8 ~18 GB)
#   .tools/hf             modelos HuggingFace
# Para remover tudo: rm -rf .tools .venv
#
# Uso: bash scripts/setup.sh [--yes] [--skip-model] [--skip-ollama] [--skip-hf]
set -euo pipefail

UV_VERSION="0.13.7"
PYTHON_VERSION="3.11"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
UV_DIR="$TOOLS/uv"
UV="$UV_DIR/uv"
cd "$ROOT"

export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_CACHE_DIR="$TOOLS/uv-cache"
export UV_PYTHON_PREFERENCE="only-managed"

echo "==> Projeto: $ROOT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERRO: '$1' é necessário e não foi encontrado." >&2; exit 1; }; }
need curl
need tar

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# ------------------------------------------------------------------
# 1. uv portátil
# ------------------------------------------------------------------
if [ ! -x "$UV" ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)          ASSET="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
    Linux-aarch64)         ASSET="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
    Darwin-arm64)          ASSET="uv-aarch64-apple-darwin.tar.gz" ;;
    Darwin-x86_64)         ASSET="uv-x86_64-apple-darwin.tar.gz" ;;
    *) echo "Plataforma não suportada: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
  BASE="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  mkdir -p "$UV_DIR"
  echo "==> Baixando uv $UV_VERSION ($ASSET)"
  curl -fsSL "$BASE/$ASSET" -o "$UV_DIR/$ASSET"
  EXPECTED="$(curl -fsSL "$BASE/$ASSET.sha256" | awk '{print $1}')"
  ACTUAL="$(sha256_of "$UV_DIR/$ASSET")"
  if [ -n "$EXPECTED" ] && [ "$EXPECTED" != "$ACTUAL" ]; then
    rm -f "$UV_DIR/$ASSET"; echo "SHA-256 do uv não confere" >&2; exit 1
  fi
  tar -xzf "$UV_DIR/$ASSET" -C "$UV_DIR" --strip-components=1
  rm -f "$UV_DIR/$ASSET"
  chmod +x "$UV"
else
  echo "==> uv já presente: $UV"
fi

# ------------------------------------------------------------------
# 2. Python 3.11 gerenciado + .venv com dependências travadas
# ------------------------------------------------------------------
echo "==> Criando/sincronizando .venv (Python $PYTHON_VERSION + dependências do uv.lock)"
"$UV" sync --locked --python "$PYTHON_VERSION"

# ------------------------------------------------------------------
# 3. Ollama portátil + modelo + modelos HF + diagnóstico
# ------------------------------------------------------------------
set +e
"$ROOT/.venv/bin/python" -m rag.tools.bootstrap "$@"
CODE=$?
set -e

cat <<'EOF'

Próximos passos:
  source .venv/bin/activate                # ativa o venv
  streamlit run app/streamlit_app.py       # interface (o Ollama sobe sozinho)
  python -m rag.pipeline "sua pergunta"    # linha de comando
EOF
exit $CODE
