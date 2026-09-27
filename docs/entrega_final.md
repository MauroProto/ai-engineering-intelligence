# Entrega final — Sistema Intelligence

La tarea compleja es responder preguntas sobre operacion, arquitectura y seguridad de un sistema tecnico a partir de un corpus de conocimiento. No se limita a una llamada a un LLM. Primero se recupera evidencia mediante RAG hibrido, luego se analiza, se valida y se produce la respuesta con sus referencias.

La implementacion integra inferencia DeepSeek asincrona, schemas Pydantic, persistencia Chroma y embeddings ONNX locales, supervisor LangGraph, checkpoints Redis, API FastAPI y trazas Phoenix. El arranque completo, el notebook y la carga de cinco consultas fueron ejecutados con el modelo real. Las versiones estan bloqueadas. `bash run.sh` levanta el laboratorio con dependencias y cache previamente disponibles; no descarga nada. La credencial se inyecta en memoria y nunca se publica.

El README contiene arquitectura, ejemplos de uso, manejo de errores, reinicio, limitaciones y fuentes tecnicas. Quince tests pasaron; combinan Redis, Chroma y embeddings reales con dobles de LLM y HTTP explicitamente identificados. La evidencia integrada esta en `evidence/`, el notebook ejecutado en `notebooks/`, las capturas originales en `screenshots/` y la revision manual en `docs/evaluacion.md`.

La API se ejecuta solamente en loopback. Este alcance local cumple la demostracion academica, pero no equivale a una plataforma multiusuario lista para Internet. Para eso faltan autenticacion, autorizacion por trabajo, limites de consumo y politica de retencion y cifrado.
