# Persistencia

Un checkpointer de LangGraph guarda el estado identificado por thread_id. Tras un reinicio del servidor, se reconstruye la conversación a partir del checkpoint almacenado en SQLite o Redis.
