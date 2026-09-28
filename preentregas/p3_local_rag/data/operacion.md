# Operación y persistencia

Redis conserva el estado de cada tarea y sus checkpoints. Cuando el servidor se reinicia, el proceso recupera el checkpoint por thread_id y puede continuar la conversación sin perder turnos anteriores. Un resultado parcial nunca se presenta como tarea completada.
