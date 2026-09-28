# Preentrega 2 · Pipeline LCEL de extracción técnica

`schemas.py` define el contrato Pydantic: tecnologías presentes en el texto, criticidad baja/media/alta y un resumen técnico no vacío. `chain.py` combina `ChatPromptTemplate | model.with_structured_output(TechnicalExtraction) | validación`, ejecuta `.ainvoke()` y reintenta hasta tres veces si la salida se trunca, no se puede parsear, falta un campo o el proveedor devuelve un fallo transitorio de red o límite. Los logs indican inicio, fallos de validación y resultado, pero no incluyen respuestas remotas ni credenciales.

## Ejecutar

Se requiere Python 3.12+, `pydantic`, `langchain-core`, `langchain-openai` o `langchain-anthropic` y `python-dotenv`. Copiá `.env.example` a `.env` y configurá el proveedor deseado; no subas `.env`. Desde la raíz del repositorio:

```bash
python -m preentregas.p2_structured_pipeline.main
python -m unittest discover -s preentregas/p2_structured_pipeline -p 'test_*.py'
```

El mini-script usa un texto de ejemplo sobre FastAPI, Redis y PostgreSQL. La forma esperada es:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API quedó indisponible por una caída de conexiones concurrentes."
}
```

El JSON anterior es un ejemplo de contrato, no una salida obtenida de un proveedor en esta revisión. Las pruebas usan un modelo falso para verificar composición LCEL, reintentos y validación sin tráfico ni claves.
