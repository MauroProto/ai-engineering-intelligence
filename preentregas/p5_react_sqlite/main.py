"""Ejecución optativa con un LLM; sin credencial no hay llamada remota."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from .agent import build_graph, create_saver, run_question


async def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("No hay LLM configurado; corré las pruebas sin claves.")
        return
    from langchain_openai import ChatOpenAI

    saver = create_saver(Path(".data") / "preentrega5_checkpoints.sqlite")
    try:
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                           temperature=0, timeout=30, max_retries=1)
        graph = build_graph(model, saver)
        config = {"configurable": {"thread_id": "demo-102"}, "recursion_limit": 10}
        await run_question(graph, "¿Cuántos pedidos hizo el cliente 102 y cuál es el total?", "demo-102")
        snapshot = await graph.aget_state(config)
        trace = []
        for message in snapshot.values["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                trace.append({"kind": "tool_calls", "names": [c["name"] for c in message.tool_calls]})
            elif isinstance(message, ToolMessage):
                trace.append({"kind": "tool_result", "name": message.name, "content": message.content})
            elif isinstance(message, AIMessage):
                trace.append({"kind": "answer", "content": message.content})
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    finally:
        saver.close()


if __name__ == "__main__":
    asyncio.run(main())
