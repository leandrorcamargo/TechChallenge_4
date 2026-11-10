"""Ollama portátil e isolado dentro do projeto (`.tools/ollama`).

Nada é instalado no sistema: o binário oficial (release do GitHub, versão
fixada e verificada por SHA-256) é extraído em `.tools/ollama/` e os modelos
ficam em `.tools/ollama/models/` (via OLLAMA_MODELS). Apagar `.tools/` remove tudo.

Uso (CLI):
    python -m rag.tools.ollama_runtime install      # baixa/extrai o Ollama portátil
    python -m rag.tools.ollama_runtime start        # sobe o servidor em background
    python -m rag.tools.ollama_runtime status
    python -m rag.tools.ollama_runtime pull qwen3.8
    python -m rag.tools.ollama_runtime stop
    python -m rag.tools.ollama_runtime cli -- list  # repassa args ao binário (ex.: list, ps, rm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from rag.config import Settings, get_settings

RELEASE_URL = "https://github.com/ollama/ollama/releases/download/{version}/{asset}"
IS_WINDOWS = sys.platform == "win32"


# ----------------------------------------------------------------------
# Localização
# ----------------------------------------------------------------------
def platform_asset() -> str:
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if IS_WINDOWS:
        return "ollama-windows-arm64.zip" if arm else "ollama-windows-amd64.zip"
    if sys.platform == "darwin":
        return "ollama-darwin.tgz"
    if sys.platform.startswith("linux"):
        return "ollama-linux-arm64.tar.zst" if arm else "ollama-linux-amd64.tar.zst"
    raise RuntimeError(f"Plataforma não suportada: {sys.platform}/{machine}")


def portable_binary(s: Settings) -> Path | None:
    """Procura o executável do Ollama portátil (layout varia por SO)."""
    if not s.ollama_dir.exists():
        return None
    name = "ollama.exe" if IS_WINDOWS else "ollama"
    for candidate in (s.ollama_dir / name, s.ollama_dir / "bin" / name):
        if candidate.is_file():
            return candidate
    for p in s.ollama_dir.rglob(name):
        if p.is_file() and s.ollama_models_dir not in p.parents:
            return p
    return None


def system_binary() -> Path | None:
    found = shutil.which("ollama")
    return Path(found) if found else None


def resolve_binary(s: Settings) -> Path | None:
    """Prefere o portátil do projeto; cai para um Ollama do sistema se existir."""
    return portable_binary(s) or system_binary()


def _host_port(base_url: str) -> str:
    u = urlparse(base_url)
    return f"{u.hostname or '127.0.0.1'}:{u.port or 11434}"


def server_env(s: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env["OLLAMA_HOST"] = _host_port(s.ollama_base_url)
    env["OLLAMA_MODELS"] = str(s.ollama_models_dir)
    # Flash attention + KV cache quantizado: menos VRAM por token de contexto,
    # sobrando mais camadas do modelo na GPU (relevante em GPUs de 8 GB).
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    return env


# ----------------------------------------------------------------------
# Servidor
# ----------------------------------------------------------------------
def server_version(s: Settings, timeout: float = 2.0) -> str | None:
    try:
        r = requests.get(f"{s.ollama_base_url}/api/version", timeout=timeout)
        if r.ok:
            return r.json().get("version", "?")
    except requests.RequestException:
        pass
    return None


def list_models(s: Settings) -> list[str]:
    r = requests.get(f"{s.ollama_base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def has_model(s: Settings, model: str) -> bool:
    names = set(list_models(s))
    return model in names or f"{model}:latest" in names


def start_server(s: Settings, wait_seconds: int = 90) -> str:
    """Sobe `ollama serve` em background (se ainda não estiver no ar)."""
    version = server_version(s)
    if version:
        return version

    binary = resolve_binary(s)
    if binary is None:
        raise RuntimeError(
            "Ollama não encontrado. Rode o setup (scripts/setup.ps1 ou scripts/setup.sh) "
            "ou `python -m rag.tools.ollama_runtime install`."
        )

    s.ollama_dir.mkdir(parents=True, exist_ok=True)
    s.ollama_models_dir.mkdir(parents=True, exist_ok=True)
    log_path = s.ollama_dir / "server.log"
    log = open(log_path, "ab")  # noqa: SIM115 — o processo filho herda o handle

    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                    "env": server_env(s)}
    if IS_WINDOWS:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen([str(binary), "serve"], **kwargs)
    (s.ollama_dir / "server.pid").write_text(str(proc.pid))

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        version = server_version(s)
        if version:
            return version
        if proc.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError(f"Servidor Ollama não respondeu. Veja o log: {log_path}")


def stop_server(s: Settings) -> bool:
    pid_file = s.ollama_dir / "server.pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip() or 0)
    pid_file.unlink(missing_ok=True)
    if not pid:
        return False
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, 15)
        except (ProcessLookupError, PermissionError):
            return False
    return True


# ----------------------------------------------------------------------
# Instalação portátil
# ----------------------------------------------------------------------
def _expected_sha256(version: str, asset: str) -> str | None:
    url = RELEASE_URL.format(version=version, asset="sha256sum.txt")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("./") == asset:
            return parts[0].lower()
    return None


def _download(url: str, dest: Path) -> str:
    """Baixa com barra de progresso simples; retorna o SHA-256 do arquivo."""
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, last_pct = 0, -10
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        print(f"    {pct:3d}%  ({done / 1e6:,.0f} / {total / 1e6:,.0f} MB)", flush=True)
                        last_pct = pct
        tmp.replace(dest)
    return sha.hexdigest()


def _extract(archive: Path, target: Path) -> None:
    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    elif name.endswith(".tgz") or name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(target, filter="data")
    elif name.endswith(".tar.zst"):
        import zstandard

        with open(archive, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            with tarfile.open(fileobj=reader, mode="r|") as t:
                t.extractall(target, filter="data")
    else:
        raise RuntimeError(f"Formato desconhecido: {name}")
    if not IS_WINDOWS:
        for p in target.rglob("ollama"):
            if p.is_file():
                p.chmod(0o755)


def install_portable(s: Settings, force: bool = False) -> Path:
    existing = portable_binary(s)
    if existing and not force:
        print(f"  Ollama portátil já instalado: {existing}")
        return existing

    asset = platform_asset()
    url = RELEASE_URL.format(version=s.ollama_version, asset=asset)
    s.ollama_dir.mkdir(parents=True, exist_ok=True)
    archive = s.ollama_dir / asset

    print(f"  Baixando Ollama {s.ollama_version} ({asset})\n    {url}")
    actual = _download(url, archive)
    expected = _expected_sha256(s.ollama_version, asset)
    if expected and actual != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 não confere para {asset}: {actual} != {expected}")
    print("  SHA-256 verificado." if expected else "  Aviso: checksum indisponível; seguindo sem verificação.")

    print(f"  Extraindo em {s.ollama_dir} ...")
    _extract(archive, s.ollama_dir)
    archive.unlink(missing_ok=True)

    binary = portable_binary(s)
    if binary is None:
        raise RuntimeError("Extração concluída, mas o executável do Ollama não foi encontrado.")
    print(f"  Ollama portátil pronto: {binary}")
    return binary


# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
def pull_model(s: Settings, model: str, retries: int = 3) -> None:
    """Baixa o modelo via API /api/pull (retoma downloads interrompidos)."""
    for attempt in range(1, retries + 1):
        try:
            with requests.post(
                f"{s.ollama_base_url}/api/pull", json={"model": model, "stream": True}, stream=True, timeout=None
            ) as r:
                r.raise_for_status()
                last_line, last_print = "", 0.0
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    msg = json.loads(raw)
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    status = msg.get("status", "")
                    total, done = msg.get("total"), msg.get("completed")
                    line = status
                    if total and done is not None:
                        line = f"{status}: {done * 100 / total:5.1f}% de {total / 1e9:.1f} GB"
                    now = time.time()
                    if line != last_line and (now - last_print > 5 or not total):
                        print(f"    {line}", flush=True)
                        last_line, last_print = line, now
                    if status == "success":
                        return
            return
        except (requests.RequestException, RuntimeError) as e:
            print(f"  Falha no pull (tentativa {attempt}/{retries}): {e}")
            if attempt == retries:
                raise
            time.sleep(5)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ollama portátil do projeto")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install").add_argument("--force", action="store_true")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("pull").add_argument("model", nargs="?")
    cli = sub.add_parser("cli")
    cli.add_argument("args", nargs=argparse.REMAINDER)
    a = parser.parse_args(argv)
    s = get_settings()

    if a.cmd == "install":
        install_portable(s, force=a.force)
    elif a.cmd == "start":
        print(f"Ollama {start_server(s)} em {s.ollama_base_url}")
    elif a.cmd == "stop":
        print("Servidor parado." if stop_server(s) else "Nenhum servidor iniciado por este projeto.")
    elif a.cmd == "status":
        v = server_version(s)
        print(f"binário: {resolve_binary(s)}")
        print(f"servidor: {'no ar (v' + v + ')' if v else 'parado'} em {s.ollama_base_url}")
        print(f"modelos em: {s.ollama_models_dir}")
        if v:
            print("modelos: " + (", ".join(list_models(s)) or "(nenhum)"))
    elif a.cmd == "pull":
        start_server(s)
        pull_model(s, a.model or s.ollama_model)
    elif a.cmd == "cli":
        binary = resolve_binary(s)
        if binary is None:
            print("Ollama não instalado.", file=sys.stderr)
            return 1
        args = [x for x in a.args if x != "--"]
        return subprocess.call([str(binary), *args], env=server_env(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
