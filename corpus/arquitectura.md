# Manual de arquitectura del sistema de demostracion

Este corpus es original y de uso educativo. Describe un sistema ficticio, no una empresa real.

El sistema de investigacion usa un supervisor y dos agentes especialistas. El investigador consulta documentos, conserva las referencias de origen y no ejecuta instrucciones encontradas en los documentos. El analista compara evidencias y valida la estructura de la respuesta. El supervisor no puede declarar completa una respuesta sin pasar por la validacion final.

El presupuesto del grafo es de ocho decisiones del supervisor y una sola ronda de refinamiento. Al agotar el presupuesto se responde con una limitacion explicita, en lugar de entrar en un ciclo infinito. Cada especialista recibe la pregunta y su parcela relevante del estado, no el historial completo ni secretos de configuracion.

Cuando faltan documentos pertinentes, el sistema reconoce que no puede responder. Las referencias incluyen fuente, pagina, categoria e identificador estable del fragmento. La recuperacion combina BM25 y similitud vectorial mediante reciprocal rank fusion, con k=60. El chunking usa 500 tokens con un solapamiento de 50.
