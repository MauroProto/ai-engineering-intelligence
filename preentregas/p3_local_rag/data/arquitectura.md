# Arquitectura del sistema

La API FastAPI recibe tareas y las pone en una cola Redis. Un worker asíncrono procesa cada tarea, consulta documentos técnicos mediante ChromaDB y registra el resultado. El supervisor de LangGraph decide cuándo pedir investigación o análisis.
