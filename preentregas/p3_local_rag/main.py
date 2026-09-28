"""Script de ingesta y consulta; la consulta remota es opcional."""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .rag import LocalRAG


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    folder = Path(__file__).resolve().parent
    rag = LocalRAG(folder / "data", Path(".data") / "preentrega3_chroma")
    added = await rag.ingest()
    print(f"Documentos indexados: {rag.collection.count()} fragmentos; nuevos: {added}")
    for chunk in await rag.retrieve("¿Cómo se conserva una conversación después de reiniciar?"):
        print(f"{chunk.source}: distancia={chunk.distance:.3f}")
    if args.index_only:
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("No hay proveedor configurado; la búsqueda local sí quedó probada.")
        return
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    for question in (
        "¿Cómo se conserva una conversación después de reiniciar?",
        "¿Cuál es la capital de Marte?",
    ):
        result = await rag.get_rag_response(question, model)
        print(question, result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
