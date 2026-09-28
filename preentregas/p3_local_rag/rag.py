"""Ingesta idempotente, recuperación semántica y generación fundamentada."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schemas import RAGAnswer, SourceChunk


class LocalModelUnavailable(RuntimeError):
    """El modelo ONNX no está en la caché; nunca se descarga por sorpresa."""


class CachedONNXEmbeddings(ONNXMiniLM_L6_V2):
    """Embeddings semánticos locales con política estricta de cero descargas."""

    def _download_model_if_not_exists(self):
        folder = Path(self.DOWNLOAD_PATH) / self.EXTRACTED_FOLDER_NAME
        required = (
            "config.json",
            "model.onnx",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        )
        if not all((folder / filename).is_file() for filename in required):
            raise LocalModelUnavailable("Falta el modelo ONNX local")


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un asistente técnico. Respondé exclusivamente con los fragmentos "
        "del CONTEXTO. Si no contienen la respuesta, contestá 'No lo sé según "
        "los documentos disponibles' y usá sources=[]. No inventes fuentes. "
        "Cada fuente citada debe ser uno de los nombres indicados en el contexto.",
    ),
    ("human", "PREGUNTA: {question}\n\nCONTEXTO:\n{context}"),
])


class LocalRAG:
    def __init__(
        self,
        data_dir: Path,
        db_dir: Path,
        embeddings: Any = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.embeddings = embeddings if embeddings is not None else CachedONNXEmbeddings()
        self.embedding_lock = asyncio.Lock()
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="preentrega3_mini_lm",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=500,
            chunk_overlap=50,
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with self.embedding_lock:
            vectors = await asyncio.to_thread(self.embeddings, texts)
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector)
                for vector in vectors]

    async def ingest(self) -> int:
        """Lee .md/.txt, fragmenta en 500/50 tokens y evita duplicados."""
        files = sorted([
            path for path in self.data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ])
        if not files:
            raise ValueError("La carpeta data debe contener archivos .md o .txt")
        entries: list[tuple[str, str, dict[str, str | int]]] = []
        for path in files:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            for index, chunk in enumerate(self.splitter.split_text(content)):
                item_id = hashlib.sha256(
                    f"{path.name}:{index}:{chunk}".encode("utf-8")
                ).hexdigest()[:24]
                entries.append((
                    item_id,
                    chunk,
                    {"source": path.name, "chunk_index": index},
                ))
        if not entries:
            raise ValueError("Los documentos están vacíos")
        ids = [entry[0] for entry in entries]
        existing = await asyncio.to_thread(self.collection.get, ids=ids)
        existing_ids = set(existing["ids"])
        missing = [entry for entry in entries if entry[0] not in existing_ids]
        if missing:
            vectors = await self._embed([entry[1] for entry in missing])
            await asyncio.to_thread(
                self.collection.upsert,
                ids=[entry[0] for entry in missing],
                documents=[entry[1] for entry in missing],
                metadatas=[entry[2] for entry in missing],
                embeddings=vectors,
            )
        stored = await asyncio.to_thread(self.collection.get)
        stale_ids = sorted(set(stored["ids"]) - set(ids))
        if stale_ids:
            await asyncio.to_thread(self.collection.delete, ids=stale_ids)
        return len(missing)

    async def retrieve(self, question: str, top_k: int = 4) -> list[SourceChunk]:
        if not question.strip():
            raise ValueError("La pregunta no puede estar vacía")
        if not 1 <= top_k <= 5:
            raise ValueError("top_k debe estar entre 1 y 5")
        count = await asyncio.to_thread(self.collection.count)
        if count == 0:
            raise ValueError("Primero hay que ejecutar ingest()")
        vector = (await self._embed([question]))[0]
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SourceChunk(
                id=item_id,
                source=metadata["source"],
                text=document,
                distance=distance,
            )
            for item_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    async def get_rag_response(self, question: str, model: Any) -> RAGAnswer:
        """Recupera localmente y ejecuta una cadena LCEL asíncrona."""
        chunks = await self.retrieve(question)
        context = "\n\n".join(
            f"[FUENTE {chunk.source} / {chunk.id}] {chunk.text}"
            for chunk in chunks
        )
        chain = PROMPT | model.with_structured_output(RAGAnswer)
        answer = RAGAnswer.model_validate(
            await chain.ainvoke({"question": question, "context": context})
        )
        allowed = {chunk.source for chunk in chunks}
        if not set(answer.sources).issubset(allowed):
            raise ValueError("La respuesta cita una fuente que no fue recuperada")
        if not answer.sources and not answer.answer.lower().startswith("no lo sé"):
            raise ValueError("La respuesta carece de referencias o abstención")
        return answer
