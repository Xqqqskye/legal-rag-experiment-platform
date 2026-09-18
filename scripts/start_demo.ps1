$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $projectPython) { $projectPython } else { "python" }

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    throw "缺少 .env：请先复制 .env.example 为 .env，并填写你自己的 API Key。"
}

& $pythonCommand -m server.main
