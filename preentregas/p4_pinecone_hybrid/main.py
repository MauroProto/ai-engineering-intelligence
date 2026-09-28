"""Ruta cloud opcional. Nunca se ejecuta sin decisión y credencial del alumno."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .evaluate import evaluate, load_golden_set
from .rag import PineconeStore, RAGSystem, load_chunks


async def main() -> None:
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"):
        print("No hay Pinecone configurado. Usá las pruebas locales sin claves.")
        return
    from pinecone import Pinecone, ServerlessSpec

    folder = Path(__file__).resolve().parent
    client = Pinecone()
    store = PineconeStore(
        client,
        index_name=os.getenv("INDEX_NAME", "coderhouse-ai-engineering-m4"),
        namespace=os.getenv("PINECONE_NAMESPACE", "preentrega4"),
        spec_factory=lambda: ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    system = RAGSystem(store, load_chunks(folder / "data"))
    count = await system.ingest()
    report = await evaluate(system, load_golden_set(folder / "golden_set.json"))
    print(f"Fragmentos subidos: {count}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
