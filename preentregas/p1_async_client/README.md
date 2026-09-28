# Preentrega 1 · Cliente de LLM robusto y asíncrono

Esta carpeta implementa una interfaz común para los clientes asíncronos oficiales de OpenAI y Anthropic. `AsyncLLMManager` elige el proveedor a partir de `ModelConfig`, sin ramificaciones en el código que lo usa. `ChatMessage`, `ModelConfig` y `ModelResponse` son contratos Pydantic; la temperatura está restringida a 0–2 y `max_tokens` a 1–8192.

El método `generate` espera la respuesta completa. `stream_tokens` entrega cada fragmento mediante un generador asíncrono. Las respuestas vacías, los problemas de autenticación, los límites de tasa, los timeouts y los fallos de conexión se convierten en códigos controlados. No se imprime el texto remoto del error porque podría contener información sensible. Los errores de programación desconocidos no se silencian.

## Ejecución

Se requiere Python 3.12 o posterior y los paquetes `openai`, `anthropic`, `pydantic` y `python-dotenv`. Copiá `.env.example` a `.env` y configurá solo el proveedor que quieras usar. `.env` no se versiona. Desde la raíz del repositorio:

```bash
python -m preentregas.p1_async_client.main
python -m unittest discover -s preentregas/p1_async_client -p 'test_*.py'
```

El script consulta “¿Qué es la entropía?” en modo normal y streaming para cada proveedor configurado. Si no hay ninguno, lo informa y termina sin hacer llamadas. Las pruebas incluidas usan dobles de SDK: validan el contrato y el control de errores sin tráfico ni claves, pero no se presentan como una prueba real contra los proveedores.
