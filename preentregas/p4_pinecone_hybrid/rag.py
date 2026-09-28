"""Pipeline Pinecone Serverless + BM25 local con fusión de rankings."""

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from preentregas.p3_local_rag.rag import CachedONNXEmbeddings


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    category: str
    page: int
    text: str


def load_chunks(folder: Path) -> list[DocumentChunk]:
    """Fragmenta documentos técnicos con IDs estables y metadatos trazables."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=500, chunk_overlap=50
    )
    result: list[DocumentChunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        for page, chunk in enumerate(splitter.split_text(path.read_text(encoding="utf-8")), 1):
            identifier = hashlib.sha256(
                f"{path.name}:{page}:{chunk}".encode("utf-8")
            ).hexdigest()[:24]
            result.append(DocumentChunk(identifier, path.name, path.stem, page, chunk))
    if not result:
        raise ValueError("Faltan documentos Markdown en data/")
    return result


class PineconeStore:
    """El cliente inyectable permite verificar el contrato sin credenciales."""

    def __init__(
        self,
        pc: Any,
        index_name: str,
        namespace: str,
        spec_factory: Callable[[], Any],
        embeddings: Any = None,
    ) -> None:
        self.pc = pc
        self.index_name = index_name
        self.namespace = namespace
        self.spec_factory = spec_factory
        self.embeddings = embeddings if embeddings is not None else CachedONNXEmbeddings()
        self.index = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        rows = await asyncio.to_thread(self.embeddings, texts)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in rows]

    async def ensure_index(self, dimension: int) -> None:
        if not await asyncio.to_thread(self.pc.has_index, self.index_name):
            await asyncio.to_thread(
                self.pc.create_index,
                name=self.index_name,
                vector_type="dense",
                dimension=dimension,
                metric="cosine",
                spec=self.spec_factory(),
            )
        for _ in range(30):
            description = await asyncio.to_thread(self.pc.describe_index, self.index_name)
            actual_dimension = getattr(description, "dimension", dimension)
            if actual_dimension != dimension:
                raise ValueError(
                    f"Dimensión del índice {actual_dimension}, embeddings {dimension}"
                )
            status = getattr(description, "status", {})
            ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
            if ready:
                self.index = await asyncio.to_thread(self.pc.Index, host=description.host)
                return
            await asyncio.sleep(1)
        raise TimeoutError("El índice no quedó listo en 30 segundos")

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            raise ValueError("No hay fragmentos para subir")
        vectors = await self.embed([chunk.text for chunk in chunks])
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Embeddings de dimensiones inconsistentes")
        await self.ensure_index(dimension)
        records = [
            {
                "id": chunk.id,
                "values": vector,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "category": chunk.category,
                },
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for offset in range(0, len(records), 100):
            await asyncio.to_thread(
                self.index.upsert,
                vectors=records[offset:offset + 100],
                namespace=self.namespace,
            )
        return len(records)

    async def search(self, query: str, top_k: int = 5) -> list[str]:
        if self.index is None:
            raise RuntimeError("Primero hay que preparar el índice")
        vector = (await self.embed([query]))[0]
        response = await asyncio.to_thread(
            self.index.query,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else response.matches
        return [row["id"] if isinstance(row, dict) else row.id for row in matches]


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[str]:
        scores = self.index.get_scores(tokenize(query))
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].id for i in ranking if scores[i] > 0][:top_k]


class EnsembleRetriever:
    """Fusión RRF ponderada del recuperador vectorial y BM25.

    Esta clase es propia: el paquete opcional langchain-community no está instalado.
    Su comportamiento es comprobable mediante las pruebas locales.
    """

    def __init__(self, dense: PineconeStore, lexical: BM25Retriever) -> None:
        self.dense = dense
        self.lexical = lexical

    async def search(self, query: str, top_k: int = 5) -> list[str]:
        dense_ids, lexical_ids = await asyncio.gather(
            self.dense.search(query, top_k),
            asyncio.to_thread(self.lexical.search, query, top_k),
        )
        scores: dict[str, float] = {}
        for weight, ranking in ((0.65, dense_ids), (0.35, lexical_ids)):
            for position, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + weight / (60 + position)
        return [
            identifier for identifier, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[:top_k]
        ]


class RAGSystem:
    def __init__(self, store: PineconeStore, chunks: list[DocumentChunk]) -> None:
        self.store = store
        self.chunks = chunks
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.retriever = EnsembleRetriever(store, BM25Retriever(chunks))

    async def ingest(self) -> int:
        return await self.store.upsert_chunks(self.chunks)

    async def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not 1 <= top_k <= 5:
            raise ValueError("top_k debe estar entre 1 y 5")
        ids = await self.retriever.search(query, top_k)
        return [self.by_id[identifier] for identifier in ids if identifier in self.by_id]
