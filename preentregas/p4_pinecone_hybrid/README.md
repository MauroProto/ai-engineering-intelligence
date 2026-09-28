# Preentrega 4 · RAG híbrido con Pinecone Serverless

`rag.py` fragmenta documentos Markdown en 500 tokens con 50 de solapamiento. Cada vector incluye texto, fuente, página y categoría en sus metadatos. `PineconeStore` comprueba o crea un índice serverless de dimensión igual a la del modelo ONNX local (384 para `all-MiniLM-L6-v2`), espera a que esté listo y sube fragmentos con IDs estables a un namespace. La búsqueda densa se combina con BM25 mediante fusión recíproca de rankings ponderada en `EnsembleRetriever`, una implementación propia que no requiere `langchain-community`.

El wrapper de embeddings verifica que el modelo ONNX ya esté en caché y falla sin descargarlo si falta. Esta carpeta es independiente de las otras preentregas.

`evaluate.py` ejecuta cinco preguntas con fuentes relevantes previamente etiquetadas y calcula Precision@5 y Recall@5. El corpus incluye diez documentos, por lo que recuperar cinco no garantiza encontrar la fuente correcta. Cada pregunta tiene una única fuente relevante; con cinco resultados, la Precision@5 máxima por caso es 0,20. No se debe inflar esa cifra ni confundirla con la exactitud de una respuesta generada.

`report_offline.json` conserva el resultado local verificable: las cinco fuentes relevantes aparecieron en primera posición y la media de Precision@5 fue 0,20 y de Recall@5 fue 1,00. Ese reporte usa un índice simulado en memoria; no acredita la creación de un índice Pinecone real.

## Reproducir el índice

El código sigue el contrato de [creación de índices densos serverless](https://docs.pinecone.io/guides/index-data/create-an-index) y [upsert de vectores con metadatos](https://docs.pinecone.io/guides/index-data/upsert-data) de Pinecone. Para ejecutarlo en una cuenta propia se requiere Python 3.12+, `pinecone`, `python-dotenv`, `langchain-text-splitters`, `rank-bm25`, `chromadb`, `tiktoken` y una configuración de Pinecone en `.env` (partir de `.env.example`). El archivo `.env` nunca se sube. Se usa un embedding ONNX local de 384 dimensiones en vez de OpenAI, por lo que no se necesita una segunda clave. Desde la raíz:

```bash
python -m preentregas.p4_pinecone_hybrid.main
python -m unittest discover -s preentregas/p4_pinecone_hybrid -p 'test_*.py'
```

En esta revisión no se usó ninguna credencial ni se creó un índice Pinecone. Las pruebas locales inyectan un índice falso y verifican el contrato de creación, metadatos, fusión híbrida y cálculo de métricas con embeddings ONNX reales. Esto NO prueba persistencia ni tiempos en Pinecone cloud; un reporte obtenido con el índice falso no se presenta como resultado remoto.
