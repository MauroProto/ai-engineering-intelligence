# Preentregas 1 a 5

Cada carpeta contiene el código, las instrucciones y las pruebas de una consigna distinta. No se considera enviada ninguna carpeta hasta verificar el enlace y el estado en Coderhouse.

La [preentrega 1](p1_async_client) implementa clientes asíncronos de LLM; la [preentrega 2](p2_structured_pipeline), extracción técnica validada con LCEL; la [preentrega 3](p3_local_rag), un RAG local con Chroma; la [preentrega 4](p4_pinecone_hybrid), ingesta y recuperación híbrida para Pinecone; y la [preentrega 5](p5_react_sqlite), un agente LangGraph con memoria SQLite.

Desde la raíz se pueden ejecutar todas las pruebas locales con `python -m unittest discover -s preentregas -p 'test_*.py'`. Esas pruebas no hacen llamadas a proveedores externos. Los README individuales distinguen expresamente qué se verificó con servicios reales, qué se probó con dobles y qué sigue sin probarse por falta de credenciales.
