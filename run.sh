#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Aislamiento solo del proceso hijo; no cambia el entorno del usuario.
unset OPENAI_API_KEY ANTHROPIC_API_KEY OLLAMA_API_KEY LANGSMITH_API_KEY
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-coderhouse-deepseek-verificado-2026-09-26}"
command -v redis-server >/dev/null || { echo "Instalar Redis (brew install redis / apt install redis-server)." >&2; exit 1; }
test -x .venv/bin/python || { echo "Falta el entorno de dependencias. Este script no descarga nada." >&2; exit 1; }
# No se instala ni descarga nada. Se verifica exclusivamente DeepSeek y la
# cache ONNX existente. La clave llega por el entorno de una sesion en memoria.
.venv/bin/python -c 'import asyncio
from app.llm import ModelGateway
async def check():
 gateway=ModelGateway()
 try: await gateway.validate_environment()
 finally: await gateway.close()
asyncio.run(check())'
mkdir -p .data/redis .data/phoenix
env -u DEEPSEEK_API_KEY redis-server --bind 127.0.0.1 --port 6389 --appendonly yes --dir "$PWD/.data/redis" >.data/redis.log 2>&1 &
redis_pid=$!
env -u DEEPSEEK_API_KEY PHOENIX_WORKING_DIR="$PWD/.data/phoenix" PHOENIX_HOST=127.0.0.1 PHOENIX_PORT=6006 .venv/bin/python -m phoenix.server.main serve >.data/phoenix.log 2>&1 &
phoenix_pid=$!
api_pid=""
cleanup(){
  if [[ -n "$api_pid" ]]; then
    kill "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
  fi
  kill "$redis_pid" "$phoenix_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
.venv/bin/python -c 'import time,socket
for port in [6389,6006]:
 for _ in range(100):
  with socket.socket() as sock:
   if sock.connect_ex(("127.0.0.1",port))==0: break
  time.sleep(.2)
 else: raise SystemExit(f"No inicio el servicio {port}; revisar .data/*.log")'
env -u DEEPSEEK_API_KEY .venv/bin/python -m scripts.phoenix_pricing
exec_cmd=(.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8091 --no-access-log)
"${exec_cmd[@]}" &
api_pid=$!
wait "$api_pid"
