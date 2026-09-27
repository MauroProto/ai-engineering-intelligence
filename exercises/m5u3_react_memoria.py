"""Ciclo ReAct con memoria entre turnos y herramienta de clima de demostración.

MemorySaver conserva checkpoints mientras vive el proceso. No permite recuperar
un reinicio del programa; para eso se necesita SQLite, Redis u otro backend durable.
"""

import asyncio
import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    location: str = Field(min_length=1, max_length=120, description="Ciudad mencionada por el usuario")


@tool(args_schema=WeatherInput)
async def get_weather(location: str) -> str:
    """Devuelve clima SIMULADO para la ubicación; no consulta un pronóstico real."""
    return json.dumps({"location": location, "conditions": "Soleado, 22°C",
                       "simulated": True}, ensure_ascii=False)


def build_app(llm=None, checkpointer=None):
    tools = [get_weather]
    model = (llm or ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4.1-mini"), temperature=0,
        timeout=20, max_retries=2)).bind_tools(tools)
    tool_node = ToolNode(tools)

    async def call_model(state: AgentState):
        instruction = SystemMessage(content=(
            "Sos un asistente climatológico de demostración. Usá get_weather "
            "cuando necesites clima. La herramienta devuelve datos simulados; "
            "aclaralo y no los presentes como pronóstico actual. Recordá la ciudad "
            "del historial para preguntas posteriores. Si no hay ciudad, pedila, "
            "no la inventes. Después de observar un resultado suficiente respondé "
            "sin repetir la herramienta. No publiques razonamiento interno."))
        async with asyncio.timeout(30):
            response = await model.ainvoke([instruction, *state["messages"]])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)


async def main():
    app = build_app()
    thread_config = {"configurable": {"thread_id": "dev_test_01"}, "recursion_limit": 12}
    first = await app.ainvoke({"messages": [HumanMessage(
        content="Estoy en Buenos Aires. ¿Cómo está el clima?")]}, thread_config)
    print("Primer turno:", first["messages"][-1].content)
    # Se aporta solo el nuevo mensaje. El historial lo restaura el checkpointer.
    second = await app.ainvoke({"messages": [HumanMessage(
        content="¿Y en esa misma ciudad necesito llevar abrigo?")]}, thread_config)
    print("Segundo turno:", second["messages"][-1].content)
    snapshot = await app.aget_state(thread_config)
    print("Mensajes conservados:", len(snapshot.values["messages"]))


if __name__ == "__main__":
    asyncio.run(main())
