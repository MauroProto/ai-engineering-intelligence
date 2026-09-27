# Evaluacion manual y lectura de trazas

Mauro Tomas Proto Cassina. Revision del 26 de septiembre de 2026.

## Indice

Contexto, metodo, resultados de cinco consultas, latencia y costo, calidad,
mejora de eficiencia, persistencia y aprobacion humana, limites y fuentes.

## Contexto y metodo

Se uso un corpus original de cuatro documentos educativos, no informacion de
una empresa real. El investigador recupera documentos con Chroma y BM25 y el
analista compara los aportes. El supervisor usa DeepSeek Flash y una politica
determinista exige investigacion, analisis y validacion antes de terminar.
Las herramientas tienen contratos Pydantic y dejan spans en Phoenix.

Se enviaron cinco solicitudes HTTP concurrentes. Una pregunta financiera fue
introducida intencionalmente fuera del alcance del corpus. Se registro como
una abstencion controlada, no como una caida tecnica: el job termina COMPLETED
con answerable=false. No se presenta el estado UNSET de OpenTelemetry como un
ERROR. Las excepciones tecnicas a FAILED se verifican por separado en tests.

Las respuestas completas, fuentes, decisiones operativas y tokens del proveedor
se conservan en evidence/benchmark.json. Las capturas originales estan en
screenshots/. El proyecto Phoenix de esta muestra tiene cinco trazas raiz y
esta separado de pruebas anteriores y del notebook.

## Resultados y revision de groundedness

La consulta sobre ciclos y aportes responde con ocho decisiones y un solo
refinamiento, y explica que la validacion final no puede omitirse. Esta
respaldada por arquitectura.md y seguridad.md. Es una descripcion del corpus;
la comprobacion tecnica adicional esta en app/graph.py y los tests.

La consulta sobre reinicio identifica Redis y thread_id=job_id, y distingue
reiniciar FastAPI de conservar Redis. Esta respaldada por operacion.md y
seguridad.md. El resultado no garantiza ausencia de perdida ante toda caida:
Redis AOF con fsync cada segundo tiene una ventana de perdida abrupta.

La consulta sobre rechazo humano indica CANCELLED y ausencia de ejecucion de
la accion. La fuente seguridad.md respalda esa conclusion. La politica no
realiza envios, compras ni cambios externos; la accion es una demostracion.

La consulta sobre p95 y costo usa contadores usage, diferencia cache de entrada
no cacheada y salida, y explica la interpolacion lineal. Esta fundamentada en
metricas.md. Los precios son una fotografia OFF-PEAK verificada esa fecha,
no tarifas universales ni una factura.

La pregunta sobre facturacion anual de 2025 no tiene respuesta en estos
documentos. El sistema lo reconoce y no confunde las tarifas del proveedor
con ingresos empresariales. Esta abstencion es el resultado esperado. En esta
revision manual no se detecto una afirmacion sustantiva inventada en las cinco
respuestas; no equivale a certificar una tasa general de alucinaciones.

## Latencia, tokens y costo

La muestra enviada a las 00:36 UTC del 27/09 (21:36 del 26/09 en Argentina)
tuvo p95 de 7,571162 s desde encolado a finalizacion. Phoenix muestra 7,57 s
para sus trazas de worker. La pequena diferencia se debe a espera de cola y
limites temporales distintos. Se guardan valores exactos en los JSON.

Hubo 25 llamadas LLM, 22.271 tokens de entrada y 2.808 de salida. El costo
total estimado fue 0,003557802 USD; cada ejecucion costo menos de 0,01 USD,
por lo que Phoenix redondea sus celdas a <$0.01. Esa etiqueta no significa
costo cero. Los contadores de cache se restan de la entrada no cacheada para
no facturarlos dos veces. Los embeddings ONNX no facturan API ni tienen un
contador compatible medido; no se inventan tokens para ellos.

El proceso mas largo fue la pregunta sobre ciclos del supervisor: 7,613446 s,
5.155 tokens LLM y 0,000817092 USD estimados. Su analista consumio 2.371 tokens.
En toda la muestra, el span individual de llamada LLM mas lento duro 2,036880 s,
mientras la recuperacion mas lenta duro 0,238515 s. El mayor consumo individual
fue el analista de la pregunta de metricas, con 2.456 tokens.

El span HTTP del LLM combina envio de red, procesamiento del proveedor y
recepcion. No permite separar esas partes sin telemetria del servidor remoto;
no se llama a todo ese tiempo 'latencia de red'. En cambio, la recuperacion
instrumenta embeddings locales, busqueda Chroma, BM25 y fusion, y la validacion
Pydantic es procesamiento local. Los spans anidados no deben sumarse porque
duplican tiempos de sus padres.

## Observacion concreta de eficiencia

Cada trabajo usa tres decisiones LLM del supervisor y dos llamadas de los
especialistas. La latencia acumulada de las decisiones puede superar la de
una sola llamada del analista. Una mejora a evaluar seria usar reglas para
los pasos obligatorios y llamar al supervisor solo ante ambiguedad o conflicto,
o reducir contexto redundante del analista sin eliminar fuentes. No se cambio
esa arquitectura durante la muestra ni se afirma un ahorro ya medido. Una
comparacion posterior debe conservar preguntas, corpus, tarifa y metodologia.

## Persistencia, aprobacion humana y pruebas

Quince tests pasaron con Redis, Chroma y embeddings reales. Los LLM/HTTP de
los tests usan dobles explicitamente identificados. La carga y el notebook
son las ejecuciones reales del modelo. Se prueba restauracion con un saver
y un grafo nuevos, estados concurrentes independientes y paso a FAILED.

evidence/hitl.json conserva una corrida anterior y separada de aprobacion y
rechazo real por endpoint. Incluye una version anterior del texto del corpus;
esa referencia no describe la configuracion actual del proveedor. La prueba
acredita WAITING_APPROVAL, COMPLETED al aprobar, CANCELLED al rechazar, 409
al duplicar la aprobacion y 422 para entrada invalida. No se mezcla con p95.

## Limites y fuentes

Son cinco consultas de un laboratorio educativo. No constituyen una muestra
estadistica estable ni un SLO. La validacion de referencias y esquemas no
garantiza verdad semantica. Chroma es local; no se afirma integracion Pinecone.
La API y Phoenix se enlazan a loopback y no estan listas para usuarios externos
sin autenticacion, autorizacion, cifrado, retencion y limites de consumo.

La clave autorizada no se incluye en documentos, codigo, notebooks ni trazas.
Se requiere una credencial propia inyectada al proceso para reproducir llamadas.
No se descargaron modelos ni dependencias durante esta revision.

Fuentes primarias: corpus/arquitectura.md, corpus/operacion.md,
corpus/seguridad.md, corpus/metricas.md y evidence/phoenix_metrics.json.
Precios: https://api-docs.deepseek.com/quick_start/pricing/
JSON: https://api-docs.deepseek.com/guides/json_mode/
