# Preentrega 5 · Agente cíclico con memoria SQLite

`agent.py` define un `StateGraph(MessagesState)` con nodo de modelo, `ToolNode` y arista condicional `tools_condition`. El modelo, no una serie de `if/else` sobre el texto del usuario, decide si debe llamar herramientas. Las herramientas `find_orders` y `sum_order_amounts` obligan a dos pasos para responder cantidad y total. `ToolNode` devuelve los errores al modelo para permitir corrección o aclaración. Cada ejecución tiene un límite de 10 pasos.

`sqlite_saver.py` implementa un `SqliteSaver` propio sobre la interfaz de LangGraph, usando SQLite de la biblioteca estándar. Persiste checkpoints, escrituras y blobs en transacciones; una instancia nueva recupera el historial por `thread_id` incluso tras cerrar la primera. No es la clase del paquete opcional `langgraph-checkpoint-sqlite`; esa dependencia no se instaló. Las pruebas de reinicio verifican la funcionalidad de esta implementación.

## Ejecutar

Se requiere Python 3.12+, `langgraph`, `langchain-core`, `langchain-openai`, `pydantic` y `python-dotenv`. Desde la raíz:

```bash
python -m unittest discover -s preentregas/p5_react_sqlite -p 'test_*.py'
python -m preentregas.p5_react_sqlite.main
```

El script de ejecución requiere que el usuario configure un proveedor en un `.env` local no versionado; sin él no transmite nada. Las pruebas usan un modelo falso cuyo recorrido genera dos llamadas a herramientas y un segundo turno con el mismo `thread_id`. La traza en `trace_offline.json` muestra ese recorrido local y está rotulada como simulación; no se presenta como razonamiento de un LLM remoto.
