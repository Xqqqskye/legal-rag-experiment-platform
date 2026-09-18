#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and add your own API key." >&2
  exit 1
fi

python_bin="python"
if [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi

exec "$python_bin" -m server.main
