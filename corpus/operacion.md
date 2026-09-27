# Manual operativo del sistema de demostracion

La API POST /tasks valida la consulta con Pydantic, guarda el estado QUEUED en Redis y encola el trabajo en la misma transaccion. Devuelve HTTP 202 con job_id sin esperar a que termine el modelo. GET /tasks/{job_id} permite consultar QUEUED, RUNNING, WAITING_APPROVAL, COMPLETED, CANCELLED o FAILED.

Los workers atienden hasta cinco trabajos concurrentes. Las excepciones y los timeouts cambian el trabajo a FAILED. Cada ejecucion tiene un limite configurable de hasta 600 segundos, por defecto 600 para inferencia local. Cada llamada al modelo tiene su propio timeout de 120 segundos. Una tarea lenta no bloquea el event loop. Una cola de procesamiento conserva los trabajos retirados hasta su confirmacion. Al reiniciar un worker aislado recupera los trabajos RUNNING pendientes desde el ultimo checkpoint.

Los checkpoints de LangGraph se guardan en Redis con thread_id igual a job_id. Redis debe usar appendonly yes para sobrevivir a su propio reinicio. Reiniciar solamente FastAPI no borra los resultados ni las conversaciones.
