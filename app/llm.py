import asyncio
import json
import os
from pathlib import Path
import httpx
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pydantic import ValidationError
from .config import settings
from .observability import tracer, span_input, span_output, usage_cost


class ModelUnavailableError(RuntimeError):
    """Falta el proveedor autorizado; nunca se sustituye por otra credencial."""


class ModelResponseError(ValueError):
    """Respuesta incompleta o sin metricas reales."""


class CachedEmbeddings(ONNXMiniLM_L6_V2):
    """Inferencia ONNX real. No descarga nada ni consulta servicios externos."""
    def _download_model_if_not_exists(self):
        folder = Path(self.DOWNLOAD_PATH) / self.EXTRACTED_FOLDER_NAME
        required = ("config.json", "model.onnx", "special_tokens_map.json",
                    "tokenizer_config.json", "tokenizer.json", "vocab.txt")
        if not all((folder / name).is_file() for name in required):
            raise ModelUnavailableError("Falta el modelo ONNX local de embeddings; no se descarga automaticamente")


class ModelGateway:
    """Solo DeepSeek autorizado. No lee keys OpenAI/Anthropic ni registra headers."""
    def __init__(self, transport=None, api_key=None):
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        self.client = httpx.AsyncClient(
            base_url=settings.deepseek_base_url, timeout=settings.model_timeout_seconds,
            headers={"Authorization":"Bearer "+self.api_key} if self.api_key else {},
            trust_env=False, follow_redirects=False, transport=transport)
        self.semaphore = asyncio.Semaphore(settings.worker_concurrency)
        self.embedding_lock = asyncio.Lock()
        self.embedding_function = CachedEmbeddings()
        self.ready = False

    async def validate_environment(self):
        if not self.api_key:
            raise ModelUnavailableError("Falta la clave DeepSeek en la memoria del proceso; no se usan otras claves")
        response = await self.client.get("/models")
        response.raise_for_status()
        if settings.llm_model not in {row["id"] for row in response.json().get("data",[])}:
            raise ModelUnavailableError("El modelo configurado no esta disponible en DeepSeek")
        self.embedding_function._download_model_if_not_exists()
        self.ready = True

    async def structured(self, schema, system, payload):
        if not self.ready:
            await self.validate_environment()
        measurements = []
        # Tres intentos acotados; no se fabrican resultados ni se cambia de proveedor.
        for attempt in range(3):
            try:
                async with self.semaphore:
                    with tracer.start_as_current_span(f"llm.{schema.__name__}") as span:
                        span_input(span, payload, "LLM")
                        span.set_attribute("llm.model_name", settings.llm_model)
                        response = await self.client.post("/chat/completions", json={
                            "model":settings.llm_model,
                            "messages":[{"role":"system", "content":system + "\nResponde solo JSON segun este esquema: " + json.dumps(schema.model_json_schema())},
                                        {"role":"user", "content":json.dumps(payload, ensure_ascii=False)}],
                            "response_format":{"type":"json_object"}, "stream":False,
                            "thinking":{"type":"disabled"}, "temperature":0, "max_tokens":1500,
                        })
                        response.raise_for_status()
                        data = response.json()
                        raw_usage=data.get("usage",{})
                        counters = {"input_tokens":raw_usage.get("prompt_tokens"),
                                    "output_tokens":raw_usage.get("completion_tokens"),
                                    "cached_tokens":raw_usage.get("prompt_cache_hit_tokens",0)}
                        if not all(type(v) is int and v >= 0 for v in counters.values()):
                            raise ModelResponseError("DeepSeek no informo contadores validos de tokens")
                        measurements.append(counters)
                        attempt_usage={"kind":"chat","provider":"deepseek",**counters}
                        span.set_attribute("llm.token_count.prompt",counters["input_tokens"])
                        span.set_attribute("llm.token_count.completion",counters["output_tokens"])
                        span.set_attribute("llm.token_count.prompt_details.cache_read",counters["cached_tokens"])
                        span.set_attribute("llm.cost.total",usage_cost(attempt_usage))
                        span.set_attribute("llm.provider","deepseek")
                        choice=data["choices"][0]
                        if choice.get("finish_reason") != "stop" or not choice["message"].get("content"):
                            raise ModelResponseError("Respuesta incompleta del modelo autorizado")
                        result = schema.model_validate_json(choice["message"]["content"])
                        usage = {"kind":"chat", "model":data["model"], "provider":"deepseek",
                                 "tokens_measured":True, "attempts":len(measurements),
                                 "billing":"estimacion_usage_y_tarifas_OFF_PEAK_2026-09-26; no_factura",
                                 **{key:sum(row[key] for row in measurements) for key in counters}}
                        usage["cost_usd"] = usage_cost(usage)
                        span_output(span,result.model_dump())
                        return result, usage
            except (httpx.TransportError, ValidationError, ModelResponseError):
                if attempt == 2:
                    raise
                await asyncio.sleep(.5 * 2**attempt)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (429,500,502,503,504) or attempt == 2:
                    raise
                await asyncio.sleep(.5 * 2**attempt)

    async def embeddings(self, texts):
        # La inferencia ONNX va fuera del event loop. Un lock evita concurrencia
        # sobre el tokenizer mutable. No usa keys ni downloads.
        async with self.embedding_lock:
            vectors = await asyncio.to_thread(self.embedding_function, texts)
        usage = {"kind":"embedding", "model":settings.embedding_model,
                 "provider":"onnx_local", "tokens_measured":False,
                 "input_tokens":0, "output_tokens":0, "cached_tokens":0,
                 "billing":"sin_facturacion_API; no_se_inventan_tokens_de_embeddings"}
        usage["cost_usd"] = usage_cost(usage)
        return [row.tolist() for row in vectors], usage

    async def close(self):
        await self.client.aclose()
        self.api_key=None
