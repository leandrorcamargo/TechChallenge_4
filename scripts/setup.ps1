<#
.SYNOPSIS
    Prepara TODO o ambiente do projeto no Windows, de forma isolada (estilo venv).

.DESCRIPTION
    Nada é instalado no sistema. Tudo fica dentro do projeto:
      .tools\uv        -> uv (gerenciador de Python/venv), versão fixada
      .tools\python    -> Python 3.11 gerenciado pelo uv
      .venv            -> ambiente virtual com as dependências travadas (uv.lock)
      .tools\ollama    -> Ollama portátil (binário oficial, SHA-256 verificado)
      .tools\ollama\models -> modelos LLM (qwen3.8 ~18 GB)
      .tools\hf        -> modelos HuggingFace (reranker / embedding)
    Para remover tudo: apague .tools\ e .venv\.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Yes
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipModel   # sem o download de 18 GB
#>
[CmdletBinding()]
param(
    [switch]$Yes,         # não pergunta; aceita os downloads
    [switch]$SkipModel,   # não baixa o LLM
    [switch]$SkipOllama,  # não instala/inicia o Ollama
    [switch]$SkipHF       # não pré-baixa modelos HuggingFace
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest fica muito mais rápido
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$UvVersion     = '0.13.7'
$PythonVersion = '3.11'

$Root  = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root '.tools'
$UvDir = Join-Path $Tools 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
Set-Location $Root

# uv guarda o Python e o cache dentro do projeto (isolamento total)
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
$env:UV_CACHE_DIR          = Join-Path $Tools 'uv-cache'
$env:UV_PYTHON_PREFERENCE  = 'only-managed'
$env:PYTHONUTF8            = '1'

Write-Host "==> Projeto: $Root"

# ------------------------------------------------------------------
# 1. uv portátil
# ------------------------------------------------------------------
if (-not (Test-Path $UvExe)) {
    $arch  = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $asset = "uv-$arch-pc-windows-msvc.zip"
    $base  = "https://github.com/astral-sh/uv/releases/download/$UvVersion"
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $zip = Join-Path $UvDir $asset

    Write-Host "==> Baixando uv $UvVersion ($asset)"
    Invoke-WebRequest -Uri "$base/$asset" -OutFile $zip -UseBasicParsing
    # No PowerShell 5.1 o .Content de um .sha256 chega como byte[]: decodificar antes.
    $shaContent = (Invoke-WebRequest -Uri "$base/$asset.sha256" -UseBasicParsing).Content
    if ($shaContent -is [byte[]]) { $shaContent = [Text.Encoding]::ASCII.GetString($shaContent) }
    $expected = ($shaContent.Trim() -split '\s+')[0].ToLower()
    $actual   = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($actual -ne $expected) {
        Remove-Item $zip -Force
        throw "SHA-256 do uv não confere ($actual != $expected)"
    }
    Expand-Archive -Path $zip -DestinationPath $UvDir -Force
    Remove-Item $zip -Force
    $found = Get-ChildItem -Path $UvDir -Recurse -Filter 'uv.exe' | Select-Object -First 1
    if (-not $found) { throw "uv.exe não encontrado após extração" }
    if ($found.FullName -ne $UvExe) { Copy-Item $found.FullName $UvExe -Force }
} else {
    Write-Host "==> uv já presente: $UvExe"
}

# ------------------------------------------------------------------
# 2. Python 3.11 gerenciado + .venv com dependências travadas (uv.lock)
# ------------------------------------------------------------------
Write-Host "==> Criando/sincronizando .venv (Python $PythonVersion + dependências do uv.lock)"
# Retry: no Windows o antivírus às vezes trava arquivos recém-baixados no cache
# ("Access is denied (os error 5)"); uma nova tentativa resolve.
$synced = $false
foreach ($attempt in 1..3) {
    & $UvExe sync --locked --python $PythonVersion
    if ($LASTEXITCODE -eq 0) { $synced = $true; break }
    Write-Host "    uv sync falhou (tentativa $attempt/3); tentando de novo em 5s..."
    Start-Sleep -Seconds 5
}
if (-not $synced) { throw "uv sync falhou após 3 tentativas" }

$Py = Join-Path $Root '.venv\Scripts\python.exe'

# ------------------------------------------------------------------
# 3. Ollama portátil + modelo + modelos HF + diagnóstico (Python, multiplataforma)
# ------------------------------------------------------------------
$bootArgs = @()
if ($Yes)        { $bootArgs += '--yes' }
if ($SkipModel)  { $bootArgs += '--skip-model' }
if ($SkipOllama) { $bootArgs += '--skip-ollama' }
if ($SkipHF)     { $bootArgs += '--skip-hf' }

& $Py -m rag.tools.bootstrap @bootArgs
$code = $LASTEXITCODE

Write-Host ""
Write-Host "Próximos passos:"
Write-Host "  .venv\Scripts\Activate.ps1                 # ativa o venv"
Write-Host "  streamlit run app\streamlit_app.py         # interface (o Ollama sobe sozinho)"
Write-Host "  python -m rag.pipeline `"sua pergunta`"     # linha de comando"
exit $code
