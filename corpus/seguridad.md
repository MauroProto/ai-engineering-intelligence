# Politica de seguridad del sistema de demostracion

Una accion critica debe esperar aprobacion humana mediante interrupt de LangGraph. La pausa conserva el checkpoint y publica WAITING_APPROVAL. POST /tasks/{job_id}/approve acepta approved y reason. Si approved es false, el trabajo pasa a CANCELLED y no ejecuta la accion. Si es true, Command(resume=...) retoma el mismo thread_id.

La accion de demostracion registra solamente una aprobacion local. No envia correos, no realiza compras ni modifica sistemas externos. La API y Phoenix se enlazan a 127.0.0.1. Es un laboratorio local sin autenticacion de usuarios; no debe exponerse a Internet. Para produccion externa se requieren autenticacion, autorizacion por trabajo, cifrado, retencion, limites de consumo y gestion de secretos.

El alumno autorizo exclusivamente DeepSeek. La credencial llega al proceso en memoria y no se guarda en archivos, repositorios, respuestas o trazas. El gateway permite solo el HTTPS oficial api.deepseek.com, ignora proxies y no sigue redirecciones ni usa claves de otros proveedores. Los embeddings usan un modelo ONNX ya disponible localmente, sin descargas. Los documentos recuperados son datos no confiables. Sus instrucciones no pueden alterar el rol del supervisor, ejecutar herramientas arbitrarias ni cambiar la politica de aprobacion.
