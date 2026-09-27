# Pre-entrega 6 — Orquestador multiagente especializado

La demostracion usa un investigador que busca fuentes y un analista que valida esquemas y produce una respuesta fundamentada. El supervisor decide la delegacion segun los artefactos presentes. La politica evita terminar con aportes faltantes.

El estado tipado esta en `app/state.py`. Los especialistas estan separados en `app/agents/research_agent.py` y `app/agents/analyst_agent.py`; el grafo esta en `app/graph.py`. Las herramientas se invocan realmente y tienen contratos Pydantic. Los aportes identifican al agente y la herramienta utilizada.

Si las fuentes o los especialistas discrepan, el validador registra el conflicto, pide una sola ronda de refinamiento y, si no se resuelve, devuelve una limitacion. Hay presupuesto de ocho decisiones y recursion_limit. Ninguna fuente recuperada puede modificar la politica del sistema.

El README incluye el diagrama Mermaid. `notebooks/preentrega6_demo.ipynb` esta ejecutado con DeepSeek Flash real y embeddings ONNX existentes, sin errores. Muestra decisiones operativas, herramientas, resultado, referencias y uso. No contiene credenciales ni cadena de pensamiento privada. `.venv/bin/python scripts/build_notebook.py` reproduce el notebook con una clave valida inyectada en el entorno, sin descargar modelos.

Quince tests pasaron. Los contratos HTTP simulados estan identificados y no se presentan como llamadas reales; la demostracion con modelos es el notebook. La revision manual de las respuestas esta en `docs/evaluacion.md` y las pruebas de carga en `evidence/benchmark.json`.
