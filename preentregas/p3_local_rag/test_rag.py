"""Pruebas de Chroma+ONNX locales y de generación con doble explícito."""

import tempfile
import unittest
from pathlib import Path

import tiktoken
from langchain_core.runnables import RunnableLambda

from preentregas.p3_local_rag.rag import LocalRAG
from preentregas.p3_local_rag.schemas import RAGAnswer


class FakeStructuredModel:
    def __init__(self, answer: str, sources: list[str]):
        self.answer = answer
        self.sources = sources
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is RAGAnswer

        def answer_prompt(prompt):
            self.prompts.append(prompt.to_messages()[-1].content)
            return {"answer": self.answer, "sources": self.sources}

        return RunnableLambda(answer_prompt)


class TestLocalRAG(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sandbox = tempfile.TemporaryDirectory(prefix="coderhouse-p3-")
        base = Path(self.sandbox.name)
        data_dir = Path(__file__).resolve().parent / "data"
        self.rag = LocalRAG(data_dir, base / "chroma")
        self.first_count = await self.rag.ingest()

    async def asyncTearDown(self):
        self.sandbox.cleanup()

    async def test_ingestion_is_idempotent_and_semantic(self):
        self.assertEqual(self.first_count, 3)
        self.assertEqual(await self.rag.ingest(), 0)
        self.assertEqual(self.rag.collection.count(), 3)
        rows = await self.rag.retrieve(
            "¿Qué mecanismo recupera una conversación tras reiniciar?", top_k=2
        )
        self.assertIn("operacion.md", {row.source for row in rows})
        self.assertTrue(all(row.text and row.id for row in rows))

    async def test_grounded_answer_and_abstention_contract(self):
        known = FakeStructuredModel(
            "Redis conserva checkpoints que permiten reanudar la conversación.",
            ["operacion.md"],
        )
        result = await self.rag.get_rag_response(
            "¿Cómo se recupera una conversación después de reiniciar?", known
        )
        self.assertEqual(result.sources, ["operacion.md"])
        self.assertIn("operacion.md", known.prompts[0])

        trap = FakeStructuredModel(
            "No lo sé según los documentos disponibles.", []
        )
        unknown = await self.rag.get_rag_response(
            "¿Cuál es la capital de Marte?", trap
        )
        self.assertEqual(unknown.sources, [])
        self.assertTrue(unknown.answer.startswith("No lo sé"))

    async def test_hallucinated_source_is_rejected(self):
        fake = FakeStructuredModel("Respuesta inventada", ["marte.md"])
        with self.assertRaisesRegex(ValueError, "fuente"):
            await self.rag.get_rag_response("¿Cuál es la capital de Marte?", fake)

    async def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.rag.retrieve("pregunta", top_k=8)

    async def test_updated_source_replaces_stale_chunk(self):
        base = Path(self.sandbox.name)
        data = base / "mutable_docs"
        data.mkdir()
        source = data / "nota.md"
        source.write_text("Redis conserva resultados temporales.", encoding="utf-8")
        mutable = LocalRAG(data, base / "mutable_chroma")
        self.assertEqual(await mutable.ingest(), 1)
        old_id = mutable.collection.get()["ids"][0]
        source.write_text("LangGraph conserva checkpoints por hilo.", encoding="utf-8")
        self.assertEqual(await mutable.ingest(), 1)
        rows = mutable.collection.get(include=["documents"])
        self.assertEqual(len(rows["ids"]), 1)
        self.assertNotEqual(rows["ids"][0], old_id)
        self.assertIn("LangGraph", rows["documents"][0])

    async def test_long_document_is_split_below_500_tokens(self):
        long_text = " ".join(f"concepto{i}" for i in range(1000))
        parts = self.rag.splitter.split_text(long_text)
        tokeniser = tiktoken.get_encoding("cl100k_base")
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(tokeniser.encode(part)) <= 500 for part in parts))
