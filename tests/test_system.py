import asyncio
import math
from uuid import uuid4
import pytest
from pydantic import ValidationError
from redis.asyncio import Redis
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command
from app.config import settings
from app.state import TaskRequest,ApprovalRequest,AnalysisArtifact,IntelligenceState,RouteDecision,ResearchArtifact
from app.checkpointer import RedisCheckpointSaver
from app.graph import build_graph
from app.hitl import approval_gate
from app.storage import JobStore,PREFIX
from app.worker import WorkerPool
from app.observability import usage_cost


class FakeGateway:
    """Doble determinista exclusivamente de tests; no existe modo fake en la API."""
    async def structured(self,schema,system,payload):
        if schema is RouteDecision:
            result=RouteDecision(next_agent="validate",reason="Test de la politica contra salto inseguro")
        elif schema is ResearchArtifact:
            result=ResearchArtifact(summary="Redis guarda checkpoints",source_ids=["doc1"],sufficient=True)
        else:
            result=AnalysisArtifact(answer="Redis persiste los checkpoints",source_ids=["doc1"],
                                    answerable=True,confidence=.9,limitations="Corpus de prueba")
        return result,{"kind":"chat","input_tokens":0,"output_tokens":0,"cached_tokens":0,"cost_usd":0}


class FakeKnowledge:
    async def search(self,query,limit):
        return [{"id":"doc1","text":"Redis guarda checkpoints","source":"test","page":1,
                 "category":"test","score":.03}],{"kind":"embedding","input_tokens":0}


def test_contract_rejects_invalid_fields():
    for kwargs in ({"query":"x"},{"query":"hola","extra":True}):
        with pytest.raises(ValidationError):
            TaskRequest(**kwargs)
    for confidence in (math.nan,1.1,-.1):
        with pytest.raises(ValidationError):
            AnalysisArtifact(answer="a",source_ids=[],answerable=False,confidence=confidence,limitations="b")


def test_cost_counts_cached_tokens_once():
    actual=usage_cost({"input_tokens":1000,"output_tokens":100,"cached_tokens":500})
    expected=(500*settings.input_price_per_million+500*settings.cached_input_price_per_million
              +100*settings.output_price_per_million)/1e6
    assert actual==pytest.approx(expected)


@pytest.mark.asyncio
async def test_supervisor_cannot_skip_specialists_or_validation():
    graph=build_graph(FakeGateway(),FakeKnowledge())
    result=await graph.ainvoke({"query":"Como persiste?","contributions":[],"usages":[],"routing_log":[]})
    assert result["final"]["validation_passed"]
    assert result["final"]["contributors"]==["analyst","researcher"]
    assert [x["selected"] for x in result["routing_log"]]==["researcher","analyst","validate"]


@pytest.mark.asyncio
async def test_two_concurrent_graphs_do_not_cross_state():
    graph=build_graph(FakeGateway(),FakeKnowledge())
    results=await asyncio.gather(*[graph.ainvoke({"query":q,"contributions":[],"usages":[],"routing_log":[]})
                                  for q in ("consulta uno","consulta dos")])
    assert [x["query"] for x in results]==["consulta uno","consulta dos"]
    assert all(len(x["contributions"])==2 for x in results)


@pytest.mark.asyncio
async def test_redis_interrupt_survives_new_saver_and_rejection():
    redis=Redis.from_url(settings.redis_url)
    thread=str(uuid4())
    saver=RedisCheckpointSaver(redis,prefix="intelligence:test:cp:"+thread)
    builder=StateGraph(IntelligenceState)
    builder.add_node("approval",approval_gate)
    builder.add_edge(START,"approval")
    builder.add_edge("approval",END)
    config={"configurable":{"thread_id":thread}}
    paused=await builder.compile(checkpointer=saver).ainvoke({"query":"Critica","critical_action":True},config)
    assert paused["__interrupt__"]
    # Nuevo objeto y grafo: nada depende de la memoria del proceso anterior.
    resumed=await builder.compile(checkpointer=RedisCheckpointSaver(redis,prefix=saver.prefix)).ainvoke(
        Command(resume={"approved":False,"reason":"No autorizado"}),config)
    assert resumed["declined"] and not resumed["approved"]
    history=[x async for x in saver.alist(config)]
    assert len(history)>=3
    assert history[0].checkpoint["channel_values"]["declined"]
    await redis.aclose()


@pytest.mark.asyncio
async def test_worker_exception_persists_failed_in_real_redis():
    redis=Redis.from_url(settings.redis_url)
    store=JobStore(redis)
    job=await store.create(TaskRequest(query="Prueba de fallo intencional del worker"))
    class BrokenGraph:
        async def aget_state(self,config):
            raise RuntimeError("Fallo intencional de test")
    pool=WorkerPool(store,BrokenGraph())
    # Consumidor aislado; no compite con API en esta prueba previa al arranque.
    task=asyncio.create_task(pool.consume())
    for _ in range(100):
        observed=await store.get(job["job_id"])
        if observed["status"]=="FAILED":
            break
        await asyncio.sleep(.02)
    task.cancel()
    await asyncio.gather(task,return_exceptions=True)
    assert observed["status"]=="FAILED"
    assert observed["error"]["type"]=="RuntimeError"
    await redis.aclose()
