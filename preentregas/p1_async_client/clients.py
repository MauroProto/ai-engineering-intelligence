"""Adaptadores de SDK asíncronos, con una interfaz común y errores seguros."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class LLMServiceError(RuntimeError):
    """Fallo esperado del proveedor, sin exponer el mensaje remoto ni secretos."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def classify_provider_error(exc: Exception) -> str | None:
    """Clasifica fallos recuperables sin registrar el cuerpo de la excepción."""
    name = type(exc).__name__
    if name in {"RateLimitError", "TooManyRequestsError"}:
        return "rate_limit"
    if name in {"APITimeoutError", "TimeoutError"}:
        return "timeout"
    if name in {"APIConnectionError", "APIError", "ServiceUnavailableError"}:
        return "network"
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and status >= 500:
        return "provider_unavailable"
    return None


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta sin bloquear el event loop."""

    @abstractmethod
    def stream_tokens(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Devuelve los fragmentos del modelo conforme llegan."""

    def _failure(self, code: str) -> ModelResponse:
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            error_code=code,
        )


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, sdk: Any = None) -> None:
        super().__init__(config)
        if config.provider is not Provider.OPENAI:
            raise ValueError("La configuración no corresponde a OpenAI")
        if sdk is None:
            from openai import AsyncOpenAI

            sdk = AsyncOpenAI()
        self.sdk = sdk

    @staticmethod
    def _messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [message.model_dump() for message in messages]

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            result = await self.sdk.chat.completions.create(
                model=self.config.model,
                messages=self._messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            choice = result.choices[0]
            text = choice.message.content or ""
            if not text.strip():
                return self._failure("empty_response")
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.config.model,
                text=text,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:
            code = classify_provider_error(exc)
            if code is None:
                raise
            return self._failure(code)

    async def stream_tokens(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.sdk.chat.completions.create(
                model=self.config.model,
                messages=self._messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                for choice in chunk.choices:
                    token = choice.delta.content
                    if token:
                        yield token
        except Exception as exc:
            code = classify_provider_error(exc)
            if code is None:
                raise
            raise LLMServiceError(code) from None


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, sdk: Any = None) -> None:
        super().__init__(config)
        if config.provider is not Provider.ANTHROPIC:
            raise ValueError("La configuración no corresponde a Anthropic")
        if sdk is None:
            from anthropic import AsyncAnthropic

            sdk = AsyncAnthropic()
        self.sdk = sdk

    @staticmethod
    def _payload(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [m.model_dump() for m in messages if m.role != "system"]
        if not turns:
            raise ValueError("Anthropic requiere al menos un turno de usuario")
        return system, turns

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        system, turns = self._payload(messages)
        try:
            result = await self.sdk.messages.create(
                model=self.config.model,
                system=system,
                messages=turns,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            text = "".join(
                block.text for block in result.content if block.type == "text"
            )
            if not text.strip():
                return self._failure("empty_response")
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                text=text,
                finish_reason=result.stop_reason,
            )
        except Exception as exc:
            code = classify_provider_error(exc)
            if code is None:
                raise
            return self._failure(code)

    async def stream_tokens(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        system, turns = self._payload(messages)
        try:
            async with self.sdk.messages.stream(
                model=self.config.model,
                system=system,
                messages=turns,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for token in stream.text_stream:
                    if token:
                        yield token
        except Exception as exc:
            code = classify_provider_error(exc)
            if code is None:
                raise
            raise LLMServiceError(code) from None
