"""Pruebas sin red del contrato de Pinecone y de la evaluación híbrida."""

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from preentregas.p4_pinecone_hybrid.evaluate import (
    evaluate,
    load_golden_set,
    precision_at_k,
    recall_at_k,
)
from preentregas.p4_pinecone_hybrid.rag import (
    PineconeStore,
    RAGSystem,
    load_chunks,
)


class FakeIndex:
    def __init__(self) -> None:
        self.records = {}
        self.upserts = []
        self.queries = []

    def upsert(self, *, vectors, namespace):
        self.upserts.append((vectors, namespace))
        for row in vectors:
            self.records[(namespace, row["id"])] = row

    def query(self, *, vector, top_k, include_metadata, namespace):
        self.queries.append((top_k, include_metadata, namespace))

        def cosine(row):
            stored = row["values"]
            norm = math.sqrt(sum(value * value for value in vector))
            stored_norm = math.sqrt(sum(value * value for value in stored))
            return sum(a * b for a, b in zip(vector, stored, strict=True)) / (norm * stored_norm)

        ranking = sorted(
            (row for (record_namespace, _), row in self.records.items()
             if record_namespace == namespace),
            key=cosine,
            reverse=True,
        )
        return {"matches": [{"id": row["id"]} for row in ranking[:top_k]]}


class FakePinecone:
    def __init__(self, dimension=None) -> None:
        self.dimension = dimension
        self.created = []
        self.index = FakeIndex()
        self.hosts = []

    def has_index(self, name):
        return self.dimension is not None

    def create_index(self, **options):
        self.created.append(options)
        self.dimension = options["dimension"]

    def describe_index(self, name):
        return SimpleNamespace(
            dimension=self.dimension,
            status={"ready": True},
            host="test-index.example.invalid",
        )

    def Index(self, *, host):
        self.hosts.append(host)
        return self.index


class TestPineconeHybrid(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = Path(__file__).resolve().parent
        self.chunks = load_chunks(self.folder / "data")
        self.fake = FakePinecone()
        self.store = PineconeStore(
            self.fake,
            index_name="test-index",
            namespace="assignment-four",
            spec_factory=lambda: {"cloud": "aws", "region": "us-east-1"},
        )
        self.system = RAGSystem(self.store, self.chunks)

    async def test_cloud_contract_and_metadata(self):
        count = await self.system.ingest()
        self.assertEqual(count, len(self.chunks))
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(self.fake.created[0]["dimension"], 384)
        self.assertEqual(self.fake.created[0]["vector_type"], "dense")
        self.assertEqual(self.fake.hosts, ["test-index.example.invalid"])
        for chunk in self.chunks:
            record = self.fake.index.records[("assignment-four", chunk.id)]
            self.assertEqual(record["metadata"], {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "category": chunk.category,
            })
        self.assertEqual(await self.system.ingest(), len(self.chunks))
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(len(self.fake.index.records), len(self.chunks))

    async def test_hybrid_retrieval_and_golden_set(self):
        await self.system.ingest()
        cases = load_golden_set(self.folder / "golden_set.json")
        report = await evaluate(self.system, cases)
        saved = json.loads((self.folder / "report_offline.json").read_text(encoding="utf-8"))
        self.assertEqual(report, {
            "cases": [
                {key: row[key] for key in (
                    "question", "retrieved_sources", "relevant_sources",
                    "precision_at_5", "recall_at_5",
                )}
                for row in saved["cases"]
            ],
            "mean_precision_at_5": saved["mean_precision_at_5"],
            "mean_recall_at_5": saved["mean_recall_at_5"],
        })
        self.assertEqual(len(report["cases"]), 5)
        self.assertEqual(report["mean_recall_at_5"], 1.0)
        self.assertEqual(report["mean_precision_at_5"], 0.2)
        self.assertTrue(all(row["precision_at_5"] == 0.2 for row in report["cases"]))
        self.assertTrue(all(request[1:] == (True, "assignment-four")
                            for request in self.fake.index.queries))

    async def test_wrong_dimension_is_rejected_before_upsert(self):
        self.fake.dimension = 1
        with self.assertRaisesRegex(ValueError, "Dimensión"):
            await self.system.ingest()
        self.assertEqual(self.fake.index.records, {})

    async def test_invalid_top_k_and_metrics(self):
        with self.assertRaises(ValueError):
            await self.system.retrieve("pregunta", top_k=6)
        self.assertEqual(precision_at_k(["a", "b"], {"b"}), 0.2)
        self.assertEqual(recall_at_k(["a", "b"], {"b", "c"}), 0.5)
        with self.assertRaises(ValueError):
            recall_at_k(["a"], set())


if __name__ == "__main__":
    unittest.main()
