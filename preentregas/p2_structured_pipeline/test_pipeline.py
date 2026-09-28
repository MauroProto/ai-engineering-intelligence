"""Pruebas LCEL sin proveedor remoto."""

import unittest
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from preentregas.p2_structured_pipeline.chain import (
    IncompleteOutputError,
    process_text,
)
from preentregas.p2_structured_pipeline.schemas import TechnicalExtraction


def valid_result(finish_reason="stop"):
    return {
        "raw": SimpleNamespace(response_metadata={"finish_reason": finish_reason}),
        "parsed": {
            "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
            "nivel_de_criticidad": "alta",
            "resumen_tecnico": "La API quedó indisponible por conexiones concurrentes.",
        },
        "parsing_error": None,
    }


class FakeModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.inputs = []

    def with_structured_output(self, schema, include_raw=False):
        assert schema is TechnicalExtraction
        assert include_raw is True

        def respond(prompt):
            self.inputs.append(prompt.to_messages()[-1].content)
            result = self.results[self.calls]
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result

        return RunnableLambda(respond)


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_lcel_async_output_is_validated(self):
        model = FakeModel([valid_result()])
        result = await process_text("FastAPI usa Redis y PostgreSQL; caída total", model)
        self.assertIsInstance(result, TechnicalExtraction)
        self.assertEqual(result.nivel_de_criticidad, "alta")
        self.assertIn("caída total", model.inputs[0])
        self.assertEqual(model.calls, 1)

    async def test_invalid_json_is_retried(self):
        invalid = {"raw": SimpleNamespace(response_metadata={}),
                   "parsed": None, "parsing_error": ValueError("mal formado")}
        model = FakeModel([invalid, valid_result()])
        result = await process_text("Redis no responde", model)
        self.assertEqual(result.tecnologias[1], "Redis")
        self.assertEqual(model.calls, 2)

    async def test_truncation_is_retried(self):
        model = FakeModel([valid_result("length"), valid_result()])
        await process_text("Error en Redis", model)
        self.assertEqual(model.calls, 2)

    async def test_three_invalid_attempts_fail_explicitly(self):
        invalid = {"raw": SimpleNamespace(response_metadata={}),
                   "parsed": None, "parsing_error": ValueError("mal formado")}
        model = FakeModel([invalid, invalid, invalid])
        with self.assertRaises(IncompleteOutputError):
            await process_text("Redis no responde", model)
        self.assertEqual(model.calls, 3)

    async def test_empty_input_is_rejected_before_model(self):
        model = FakeModel([])
        with self.assertRaises(ValueError):
            await process_text(" ", model)
        self.assertEqual(model.calls, 0)

    async def test_rate_limit_is_retried_without_exposing_provider_error(self):
        class RateLimitError(Exception):
            pass

        model = FakeModel([RateLimitError("remote body"), valid_result()])
        result = await process_text("FastAPI usa Redis; caída total", model)
        self.assertEqual(result.nivel_de_criticidad, "alta")
        self.assertEqual(model.calls, 2)

    async def test_programming_error_is_not_retried(self):
        model = FakeModel([AttributeError("bad test implementation")])
        with self.assertRaises(AttributeError):
            await process_text("FastAPI usa Redis", model)
        self.assertEqual(model.calls, 1)

    async def test_incomplete_fields_are_retried(self):
        incomplete = valid_result()
        incomplete["parsed"]["tecnologias"] = []
        model = FakeModel([incomplete, valid_result()])
        result = await process_text("FastAPI usa Redis; caída total", model)
        self.assertIn("FastAPI", result.tecnologias)
        self.assertEqual(model.calls, 2)
