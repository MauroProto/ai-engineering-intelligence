"""Pruebas del grafo y del reinicio SQLite sin llamadas a un LLM."""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from preentregas.p5_react_sqlite.agent import build_graph, create_saver, run_question


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    ])


class FakeOrderModel:
    """Doble determinista: prueba el cableado, no simula inteligencia real."""

    def __init__(self, start_with_error=False):
        self.start_with_error = start_with_error
        self.last_turn_saw_history = False

    def bind_tools(self, tools):
        assert {tool.name for tool in tools} == {"find_orders", "sum_order_amounts"}
        return self

    async def ainvoke(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            if last.content == "¿Y el último pedido?":
                self.last_turn_saw_history = any(
                    isinstance(row, ToolMessage) for row in messages[:-1]
                )
                return AIMessage(content="El último pedido registrado fue A-3.")
            if self.start_with_error:
                return tool_call("sum_order_amounts", {"order_ids": ["desconocido"]}, "error-1")
            return tool_call("find_orders", {"customer_id": 102}, "find-1")
        if isinstance(last, ToolMessage):
            if last.name == "find_orders":
                ids = json.loads(last.content)["order_ids"]
                return tool_call("sum_order_amounts", {"order_ids": ids}, "sum-1")
            if last.name == "sum_order_amounts" and "error" in last.content.lower():
                return tool_call("find_orders", {"customer_id": 102}, "retry-find")
            if last.name == "sum_order_amounts":
                data = json.loads(last.content)
                return AIMessage(
                    content=f"El cliente 102 hizo {data['orders']} pedidos por {data['total']}."
                )
        raise AssertionError("Estado inesperado en el doble")


class TestAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sandbox = tempfile.TemporaryDirectory(prefix="coderhouse-p5-")
        self.path = Path(self.sandbox.name) / "checkpoints.sqlite"
        self.saver = create_saver(self.path)

    async def asyncTearDown(self):
        self.saver.close()
        self.sandbox.cleanup()

    async def test_two_tool_calls_and_restart_preserve_session(self):
        model = FakeOrderModel()
        graph = build_graph(model, self.saver)
        first = await run_question(
            graph, "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?", "thread-102"
        )
        tool_messages = [row for row in first["messages"] if isinstance(row, ToolMessage)]
        self.assertEqual([row.name for row in tool_messages],
                         ["find_orders", "sum_order_amounts"])
        self.assertIn("3 pedidos por 14500", first["messages"][-1].content)

        self.saver.close()
        reopened = create_saver(self.path)
        self.saver = reopened
        second_model = FakeOrderModel()
        resumed = build_graph(second_model, reopened)
        second = await run_question(resumed, "¿Y el último pedido?", "thread-102")
        self.assertTrue(second_model.last_turn_saw_history)
        self.assertGreater(len(second["messages"]), len(first["messages"]))
        self.assertIn("A-3", second["messages"][-1].content)

    async def test_tool_error_can_be_corrected(self):
        graph = build_graph(FakeOrderModel(start_with_error=True), self.saver)
        result = await run_question(graph, "Consulta de cliente 102", "retry-102")
        tools = [row for row in result["messages"] if isinstance(row, ToolMessage)]
        self.assertGreaterEqual(len(tools), 3)
        self.assertIn("3 pedidos por 14500", result["messages"][-1].content)

    async def test_thread_is_required(self):
        graph = build_graph(FakeOrderModel(), self.saver)
        with self.assertRaises(ValueError):
            await run_question(graph, "consulta", "")
