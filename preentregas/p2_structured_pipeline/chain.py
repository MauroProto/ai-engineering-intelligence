"""Prompt | salida estructurada | validación, con reintentos acotados."""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .schemas import TechnicalExtraction

logger = logging.getLogger(__name__)


class IncompleteOutputError(ValueError):
    """Salida truncada, vacía o inválida que justifica reintentar."""


class TransientProviderError(RuntimeError):
    """Rate limit, timeout o indisponibilidad; nunca contiene el error remoto."""


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extraé tecnologías, criticidad y resumen del texto técnico. "
        "No agregues tecnologías ausentes. Si el texto indica caída, pérdida de "
        "datos o indisponibilidad, la criticidad es alta. Respondé según el "
        "contrato de salida estructurada provisto por el modelo.",
    ),
    ("human", "Texto técnico:\n{texto}"),
])


def validate_structured_result(result: Any) -> TechnicalExtraction:
    """Rechaza respuestas truncadas antes de convertirlas en una entrega válida."""
    if not isinstance(result, dict):
        logger.warning("Salida estructurada con formato inesperado")
        raise IncompleteOutputError("El proveedor no devolvió parsed/raw")
    raw = result.get("raw")
    metadata = getattr(raw, "response_metadata", {}) or {}
    finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if finish_reason in {"length", "max_tokens"}:
        logger.warning("Respuesta truncada por límite de tokens")
        raise IncompleteOutputError("Respuesta truncada")
    parsed = result.get("parsed")
    if result.get("parsing_error") is not None or parsed is None:
        logger.warning("Validación de salida fallida")
        raise IncompleteOutputError("JSON inválido o incompleto")
    try:
        return TechnicalExtraction.model_validate(parsed)
    except ValueError as exc:
        logger.warning("El objeto no cumple el contrato Pydantic")
        raise IncompleteOutputError("Datos incompletos") from exc


def build_chain(model: Any):
    """Inyectar un modelo permite probar LCEL sin red ni credenciales."""
    structured = model.with_structured_output(TechnicalExtraction, include_raw=True)

    async def invoke_structured(prompt: Any) -> Any:
        try:
            return await structured.ainvoke(prompt)
        except Exception as exc:
            name = type(exc).__name__
            status = getattr(exc, "status_code", None)
            transient = name in {
                "RateLimitError", "TooManyRequestsError", "APITimeoutError",
                "APIConnectionError", "ServiceUnavailableError",
            } or status == 429 or (isinstance(status, int) and status >= 500)
            if transient:
                logger.warning("Fallo transitorio del proveedor; se reintentará")
                raise TransientProviderError("Fallo transitorio del proveedor") from None
            raise

    chain = PROMPT | RunnableLambda(invoke_structured) | RunnableLambda(validate_structured_result)
    return chain.with_retry(
        retry_if_exception_type=(
            IncompleteOutputError, TransientProviderError, TimeoutError, ConnectionError,
        ),
        wait_exponential_jitter=False,
        stop_after_attempt=3,
    )


async def process_text(text: str, model: Any) -> TechnicalExtraction:
    """Ejecuta toda la cadena de forma asíncrona."""
    if not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío")
    logger.info("Iniciando extracción técnica de %d caracteres", len(text))
    try:
        result = await build_chain(model).ainvoke({"texto": text})
    except (IncompleteOutputError, TransientProviderError, TimeoutError, ConnectionError):
        logger.error("La extracción falló tras los reintentos")
        raise
    logger.info("Extracción completada con %d tecnologías", len(result.tecnologias))
    return result
