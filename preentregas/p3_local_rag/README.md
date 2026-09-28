# Preentrega 3 · RAG semántico local

El script lee los archivos `.md` y `.txt` de `data/`, los divide con `RecursiveCharacterTextSplitter` en fragmentos de hasta 500 tokens y 50 de solapamiento, y los conserva en ChromaDB con identificadores estables y nombre de fuente. Indexación y consulta usan el mismo modelo ONNX `all-MiniLM-L6-v2`, ya disponible localmente. Si el modelo falta, el programa falla de forma explícita: no descarga nada.

`LocalRAG.retrieve()` hace una búsqueda vectorial de hasta cuatro fragmentos. `get_rag_response()` usa `ChatPromptTemplate | model.with_structured_output(RAGAnswer)` y `.ainvoke()`; el prompt obliga a responder solo con ese contexto o a decir “No lo sé”. Un validador rechaza fuentes que no fueron recuperadas y respuestas sin fuente ni abstención. Ese control no demuestra por sí solo que toda frase sea verdadera.

## Ejecutar sin credenciales

Desde la raíz del repositorio, con Python 3.12+ y las dependencias indicadas en `requirements.txt`:

```bash
python -m preentregas.p3_local_rag.main --index-only
python -m unittest discover -s preentregas/p3_local_rag -p 'test_*.py'
```

La primera instrucción crea una base local ignorada por Git y comprueba recuperación semántica real. Ejecutarla dos veces no duplica los fragmentos.

Para probar generación con un LLM, copiá `.env.example` a `.env`, configurá el proveedor y ejecutá el módulo sin `--index-only`. No se incluye ninguna clave. Las pruebas automáticas de respuesta fundamentada usan un modelo falso; las pruebas de embeddings, ingesta y búsqueda sí usan el modelo ONNX local. No se presenta el doble de prueba como ejecución de un LLM remoto.
