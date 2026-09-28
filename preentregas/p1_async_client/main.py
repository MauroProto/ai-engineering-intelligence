"""Prueba manual opcional; nunca corre al importar el paquete."""

import asyncio
import os

from dotenv import load_dotenv

from .clients import LLMServiceError
from .manager import AsyncLLMManager
from .schemas import ChatMessage, ModelConfig, Provider


async def run_provider(provider: Provider, model: str) -> None:
    manager = AsyncLLMManager(ModelConfig(provider=provider, model=model))
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    result = await manager.generate(messages)
    if result.error_code:
        print(f"{provider}: error controlado ({result.error_code})")
        return
    print(f"{provider}, respuesta: {result.text}")
    print(f"{provider}, streaming: ", end="", flush=True)
    try:
        async for token in manager.stream_tokens(messages):
            print(token, end="", flush=True)
    except LLMServiceError as exc:
        print(f" [error controlado: {exc.code}]", end="")
    print()


async def main() -> None:
    load_dotenv()
    configurations = (
        (Provider.OPENAI, "OPENAI_API_KEY", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")),
    )
    available = [(provider, model) for provider, variable, model in configurations if os.getenv(variable)]
    if not available:
        print("No hay proveedores configurados; ejecutá las pruebas locales sin claves.")
        return
    for provider, model in available:
        await run_provider(provider, model)


if __name__ == "__main__":
    asyncio.run(main())
