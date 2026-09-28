"""Pruebas sin red ni credenciales para los dos adaptadores."""

import unittest
from types import SimpleNamespace

from preentregas.p1_async_client.clients import LLMServiceError
from preentregas.p1_async_client.manager import AsyncLLMManager
from preentregas.p1_async_client.schemas import ChatMessage, ModelConfig, Provider


def object_with(**values):
    return SimpleNamespace(**values)


async def fragments(*values):
    for value in values:
        yield value


class FakeOpenAI:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.chat = object_with(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return fragments(
                object_with(choices=[object_with(delta=object_with(content="Hola"))]),
                object_with(choices=[object_with(delta=object_with(content=" mundo"))]),
            )
        return object_with(choices=[
            object_with(message=object_with(content="Hola mundo"), finish_reason="stop")
        ])


class FakeAnthropicStream:
    async def __aenter__(self):
        self.text_stream = fragments("Hola", " mundo")
        return self

    async def __aexit__(self, *_):
        return False


class FakeAnthropic:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return object_with(
            content=[object_with(type="text", text="Hola mundo")],
            stop_reason="end_turn",
        )

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeAnthropicStream()


class RateLimitError(Exception):
    pass


class TestSchemas(unittest.TestCase):
    def test_limits_and_nonempty_message(self):
        for value in (-0.1, 2.1):
            with self.assertRaises(ValueError):
                ModelConfig(provider=Provider.OPENAI, model="test", temperature=value)
        with self.assertRaises(ValueError):
            ModelConfig(provider=Provider.OPENAI, model="test", max_tokens=0)
        with self.assertRaises(ValueError):
            ChatMessage(role="user", content=" ")


class TestClients(unittest.IsolatedAsyncioTestCase):
    async def test_openai_generate_and_stream(self):
        sdk = FakeOpenAI()
        manager = AsyncLLMManager(ModelConfig(provider=Provider.OPENAI, model="test"), sdk)
        messages = [ChatMessage(role="user", content="Saludá")]
        response = await manager.generate(messages)
        self.assertEqual(response.text, "Hola mundo")
        self.assertEqual(response.provider, Provider.OPENAI)
        self.assertEqual(
            "".join([part async for part in manager.stream_tokens(messages)]),
            "Hola mundo",
        )
        self.assertTrue(sdk.calls[1]["stream"])
        self.assertEqual(sdk.calls[0]["messages"], [{"role": "user", "content": "Saludá"}])

    async def test_anthropic_generate_and_stream(self):
        sdk = FakeAnthropic()
        manager = AsyncLLMManager(ModelConfig(provider=Provider.ANTHROPIC, model="test"), sdk)
        messages = [
            ChatMessage(role="system", content="Respuestas breves"),
            ChatMessage(role="user", content="Saludá"),
        ]
        response = await manager.generate(messages)
        self.assertEqual(response.text, "Hola mundo")
        self.assertEqual(
            "".join([part async for part in manager.stream_tokens(messages)]),
            "Hola mundo",
        )
        self.assertEqual(sdk.calls[0]["system"], "Respuestas breves")
        self.assertEqual(sdk.calls[0]["messages"], [{"role": "user", "content": "Saludá"}])

    async def test_rate_limit_is_controlled(self):
        sdk = FakeOpenAI(error=RateLimitError("No incluir este cuerpo en la salida"))
        manager = AsyncLLMManager(ModelConfig(provider=Provider.OPENAI, model="test"), sdk)
        messages = [ChatMessage(role="user", content="Saludá")]
        response = await manager.generate(messages)
        self.assertEqual(response.error_code, "rate_limit")
        self.assertNotIn("cuerpo", response.model_dump_json())
        with self.assertRaises(LLMServiceError) as caught:
            async for _ in manager.stream_tokens(messages):
                pass
        self.assertEqual(caught.exception.code, "rate_limit")

    async def test_programming_errors_are_not_hidden(self):
        sdk = FakeOpenAI(error=ValueError("fallo interno"))
        manager = AsyncLLMManager(ModelConfig(provider=Provider.OPENAI, model="test"), sdk)
        with self.assertRaises(ValueError):
            await manager.generate([ChatMessage(role="user", content="Saludá")])

    async def test_anthropic_rate_limit_is_controlled(self):
        sdk = FakeAnthropic(error=RateLimitError("remote error body"))
        manager = AsyncLLMManager(ModelConfig(provider=Provider.ANTHROPIC, model="test"), sdk)
        messages = [ChatMessage(role="user", content="Saludá")]
        result = await manager.generate(messages)
        self.assertEqual(result.error_code, "rate_limit")
        self.assertNotIn("remote error body", result.model_dump_json())
        with self.assertRaises(LLMServiceError) as caught:
            async for _ in manager.stream_tokens(messages):
                pass
        self.assertEqual(caught.exception.code, "rate_limit")

    async def test_network_and_server_errors_are_classified(self):
        class NetworkError(Exception):
            status_code = 503

        sdk = FakeOpenAI(error=NetworkError("internal service text"))
        manager = AsyncLLMManager(ModelConfig(provider=Provider.OPENAI, model="test"), sdk)
        result = await manager.generate([ChatMessage(role="user", content="Saludá")])
        self.assertEqual(result.error_code, "provider_unavailable")
        self.assertNotIn("internal service text", result.model_dump_json())
