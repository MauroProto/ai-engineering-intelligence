"""Mini-script de prueba; requiere un proveedor configurado por el usuario."""

import asyncio
import logging
import os

from dotenv import load_dotenv

from .chain import process_text

SAMPLE = (
    "La API de FastAPI usa Redis como caché y PostgreSQL para persistencia. "
    "Una caída de conexiones concurrentes dejó el servicio indisponible."
)


def configured_model():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), temperature=0)
    raise ValueError("LLM_PROVIDER debe ser openai o anthropic")


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if not os.getenv(f"{provider.upper()}_API_KEY"):
        print("No hay proveedor configurado. Corré las pruebas locales sin claves.")
        return
    result = await process_text(SAMPLE, configured_model())
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
