"""Fábrica y fachada utilizada por la aplicación, sin bifurcar su lógica."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from .clients import AnthropicClient, BaseLLMClient, OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, sdk: Any = None) -> None:
        clients: dict[Provider, type[BaseLLMClient]] = {
            Provider.OPENAI: OpenAIClient,
            Provider.ANTHROPIC: AnthropicClient,
        }
        self.client = clients[config.provider](config, sdk=sdk)

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    async def stream_tokens(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        async for token in self.client.stream_tokens(messages):
            yield token
