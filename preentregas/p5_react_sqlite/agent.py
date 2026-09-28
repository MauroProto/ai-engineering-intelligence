"""Agente LangGraph con decisión del modelo y herramientas especializadas."""

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field

from .sqlite_saver import SqliteSaver


ORDERS = {
    102: [
        {"id": "A-1", "amount": 4500},
        {"id": "A-2", "amount": 6000},
        {"id": "A-3", "amount": 4000},
    ],
    205: [{"id": "B-1", "amount": 2100}],
}


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(gt=0, description="Identificador numérico del cliente")


class OrderIdsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_ids: list[str] = Field(min_length=1, description="IDs devueltos por find_orders")


@tool(args_schema=CustomerInput)
async def find_orders(customer_id: int) -> str:
    """Busca los IDs de todos los pedidos de un cliente en la base de demostración."""
    await asyncio.sleep(0)
    rows = ORDERS.get(customer_id, [])
    return json.dumps({"customer_id": customer_id, "order_ids": [r["id"] for r in rows]})


@tool(args_schema=OrderIdsInput)
async def sum_order_amounts(order_ids: list[str]) -> str:
    """Suma importes de pedidos cuyos IDs se obtuvieron con find_orders."""
    await asyncio.sleep(0)
    amounts = {row["id"]: row["amount"] for rows in ORDERS.values() for row in rows}
    unknown = sorted(set(order_ids) - amounts.keys())
    if unknown:
        raise ValueError("Pedidos desconocidos: " + ", ".join(unknown))
    return json.dumps({"orders": len(order_ids),
                       "total": sum(amounts[item] for item in order_ids)})


TOOLS = [find_orders, sum_order_amounts]
SYSTEM = SystemMessage(content=(
    "Sos un asistente de pedidos de demostración. Decidí qué herramienta usar "
    "según la pregunta; no hay rutas manuales por palabras clave. Para obtener "
    "cantidad y total, primero consultá los IDs y después sumá sus importes. "
    "Si una herramienta falla, interpretá el error, corregí los argumentos o "
    "pedí aclaración. No inventes pedidos ni importes. No expongas razonamiento interno."
))


def build_graph(model: Any, checkpointer: SqliteSaver):
    bound_model = model.bind_tools(TOOLS)

    async def call_model(state: MessagesState):
        response = await bound_model.ainvoke([SYSTEM, *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer)


async def run_question(graph, question: str, thread_id: str):
    if not question.strip() or not thread_id.strip():
        raise ValueError("La pregunta y thread_id deben estar presentes")
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )


def create_saver(path: Path) -> SqliteSaver:
    return SqliteSaver(path)
