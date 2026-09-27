# Pre-entrega 7 — API de produccion y monitoreo activo

`app/main.py` expone POST /tasks, GET /tasks/{id} y POST /tasks/{id}/approve. `app/worker.py` atiende una cola Redis sin bloquear los handlers. `app/checkpointer.py` guarda los checkpoints reales de LangGraph en Redis estandar.

`app/hitl.py` pausa con interrupt las acciones criticas. El endpoint externo aprueba o rechaza y retoma con Command sobre el mismo thread_id. No hay efectos externos antes de la aprobacion. La excepcion de un worker lleva el trabajo a FAILED, no a un polling infinito.

`app/observability.py` configura Phoenix local y spans OpenInference. `scripts/benchmark.py` envio cinco peticiones concurrentes reales con DeepSeek Flash y genero `evidence/benchmark.json`, con latencias, p95, tokens medidos, costo estimado y trace_id. `evidence/hitl.json` guarda una corrida separada de aprobacion y rechazo, con 409 para aprobacion duplicada y 422 para entrada invalida. La clave solo vive en memoria; no aparece en el repositorio, resultados ni trazas. No se descargaron modelos. Los embeddings ONNX son locales y no se inventan sus tokens.

Las capturas originales en `screenshots/` muestran costo por ejecucion, latencia p95 y spans en Phoenix. Se registraron tarifas OFF-PEAK antes de ingerir los spans. El costo es una estimacion basada en usage, no una factura; no incluye hardware ni energia. El p95 de Phoenix mide la traza desde el worker; el benchmark mide encolado a finalizacion e incluye espera de cola. La diferencia pequena entre ambos esta documentada en `docs/evaluacion.md`. Cinco muestras prueban funcionalidad, no un SLO de produccion.

Se ejecuta con `bash run.sh`. No se usa memoria global de proceso como persistencia de jobs; Redis AOF conserva trabajos y checkpoints al reiniciar. Las variables de entorno estan documentadas en `.env.example` y no hay secretos publicados.
