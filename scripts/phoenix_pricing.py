"""Configura estimaciones de costos en Phoenix local; no recibe credenciales.

Las tarifas son una fotografia OFF-PEAK del 26/09/2026. Revisarlas antes de
usar el laboratorio en otro horario. Phoenix aplica tarifas al ingerir spans;
configurar antes de ejecutar el benchmark, no editar trazas historicas.
"""
import httpx


def main():
    payload = {
        "name": "DeepSeek Flash OFF-PEAK 2026-09-26",
        "provider": "deepseek",
        "namePattern": "^deepseek-flash$",
        "costs": [
            {"tokenType": "input", "costPerMillionTokens": .15, "kind": "PROMPT"},
            {"tokenType": "cache_read", "costPerMillionTokens": .003, "kind": "PROMPT"},
            {"tokenType": "output", "costPerMillionTokens": .60, "kind": "COMPLETION"},
        ],
    }
    query = "mutation($input: CreateModelMutationInput!) { createModel(input: $input) { model { id name } } }"
    response = httpx.post("http://127.0.0.1:6006/graphql", timeout=30,
                          trust_env=False, follow_redirects=False,
                          json={"query": query, "variables": {"input": payload}})
    response.raise_for_status()
    result = response.json()
    errors = result.get("errors", [])
    if errors and not all("already exists" in e.get("message", "") for e in errors):
        raise RuntimeError("No se pudieron configurar tarifas en Phoenix local")
    print("Phoenix: tarifas OFF-PEAK disponibles; no son una factura")


if __name__ == "__main__":
    main()
