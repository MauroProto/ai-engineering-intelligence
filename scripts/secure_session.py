"""Sesion interactiva. La clave se lee sin eco y vive solo en memoria de procesos.

No guarda .env, historial ni credenciales en archivos. No muestra balances ni
headers. Solo autoriza comandos conocidos y el destino HTTPS oficial DeepSeek.
"""
import getpass
import os
import subprocess
import sys
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
COMMANDS={
    "start":["bash","run.sh"],
    "benchmark":[str(ROOT/".venv/bin/python"),"-m","scripts.benchmark"],
    "notebook":[str(ROOT/".venv/bin/python"),"scripts/build_notebook.py"],
}


def main():
    key=getpass.getpass("Clave DeepSeek (entrada oculta, solo memoria): ")
    if not key:
        raise SystemExit("No se recibio una clave")
    api=None
    try:
        with httpx.Client(base_url="https://api.deepseek.com",timeout=30,
                          trust_env=False,follow_redirects=False) as client:
            response=client.get("/user/balance",headers={"Authorization":"Bearer "+key})
            if response.status_code!=200:
                raise SystemExit(f"DeepSeek: comprobacion rechazada, HTTP {response.status_code}. No se hizo una generacion.")
            if not response.json().get("is_available"):
                raise SystemExit("DeepSeek: acceso valido pero saldo no disponible para generar. No se hizo una generacion.")
            response=client.get("/models",headers={"Authorization":"Bearer "+key})
            response.raise_for_status()
            print("DeepSeek: acceso y saldo disponibles. Modelos:",
                  [row["id"] for row in response.json().get("data",[])],flush=True)
        env={k:v for k,v in os.environ.items() if k not in (
            "OPENAI_API_KEY","ANTHROPIC_API_KEY","OLLAMA_API_KEY","LANGSMITH_API_KEY")}
        env["DEEPSEEK_API_KEY"]=key
        while True:
            command=input("Sesion segura [start/benchmark/notebook/stop/exit]: ").strip()
            if command=="exit":
                break
            if command=="stop":
                if api is not None and api.poll() is None:
                    api.terminate()
                    api.wait(timeout=30)
                api=None
            elif command=="start":
                if api is not None and api.poll() is None:
                    print("La API ya esta en marcha",flush=True)
                else:
                    api=subprocess.Popen(COMMANDS[command],cwd=ROOT,env=env)
            elif command in ("benchmark","notebook"):
                result=subprocess.run(COMMANDS[command],cwd=ROOT,env=env)
                print("Comando finalizado, codigo:",result.returncode,flush=True)
            else:
                print("Comando no permitido",flush=True)
    finally:
        if api is not None and api.poll() is None:
            api.terminate()
            api.wait(timeout=30)
        key=None


if __name__=="__main__":
    main()
