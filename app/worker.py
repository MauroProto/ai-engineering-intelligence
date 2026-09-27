import asyncio
import logging
import time
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from opentelemetry import trace
from .storage import PREFIX,PENDING,PROCESSING
from .config import settings
from .observability import tracer,span_input,span_output,usage_cost

log=logging.getLogger(__name__)


class WorkerPool:
    """Un lider con N coroutines. Redis lease impide dos pools sobre la misma cola."""
    def __init__(self,store,graph):
        self.store,self.graph=store,graph
        self.owner=str(uuid4())
        self.tasks=[]
        self.lease=PREFIX+"worker-lease"

    async def start(self):
        # Tras un crash el lease vence. No recuperar trabajos de un pool activo.
        if not await self.store.redis.set(self.lease,self.owner,nx=True,ex=15):
            raise RuntimeError("Ya hay un pool activo; esperar 15 segundos si hubo un crash")
        for raw in await self.store.redis.lrange(PROCESSING,0,-1):
            job_id=raw.decode()
            job=await self.store.get(job_id)
            async with self.store.redis.pipeline(transaction=True) as pipe:
                if job and job["status"] in ("QUEUED","RUNNING"):
                    pipe.lpush(PENDING,job_id)
                pipe.lrem(PROCESSING,0,job_id)
                await pipe.execute()
        self.tasks=[asyncio.create_task(self.consume(),name=f"worker-{i}")
                    for i in range(settings.worker_concurrency)]
        self.tasks.append(asyncio.create_task(self.heartbeat(),name="lease-heartbeat"))

    async def heartbeat(self):
        script="if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('EXPIRE',KEYS[1],15) else return 0 end"
        while True:
            await asyncio.sleep(5)
            if not await self.store.redis.eval(script,1,self.lease,self.owner):
                for task in self.tasks:
                    if task is not asyncio.current_task():
                        task.cancel()
                return

    async def consume(self):
        while True:
            raw=await self.store.redis.brpoplpush(PENDING,PROCESSING,timeout=1)
            if not raw:
                continue
            job_id=raw.decode()
            job=await self.store.get(job_id)
            if not job or job["status"] not in ("QUEUED","RUNNING"):
                await self.store.redis.lrem(PROCESSING,0,job_id)
                continue
            job.update(status="RUNNING",started_at=time.time())
            await self.store.save(job)
            try:
                with tracer.start_as_current_span("task.execute") as span:
                    span_input(span,job["request"],"AGENT")
                    span.set_attribute("session.id",job_id)
                    job["trace_id"]=f"{span.get_span_context().trace_id:032x}"
                    await self.store.save(job)
                    config={"configurable":{"thread_id":job_id},"recursion_limit":40}
                    snapshot=await self.graph.aget_state(config)
                    if job.get("resume_pending"):
                        initial=Command(resume=job["approval"])
                    elif snapshot.values:
                        initial=None # Recuperar checkpoint tras interrupcion del worker.
                    else:
                        initial={**job["request"],"messages":[HumanMessage(content=job["request"]["query"])],
                                 "contributions":[],"usages":[],"routing_log":[],"steps":0}
                    async with asyncio.timeout(settings.task_timeout_seconds):
                        result=await self.graph.ainvoke(initial,config)
                    job.pop("resume_pending",None)
                    if result.get("__interrupt__"):
                        job.update(status="WAITING_APPROVAL",approval_request=result["__interrupt__"][0].value)
                    elif result.get("declined"):
                        job.update(status="CANCELLED",result={"answer":"Accion rechazada por la persona responsable"})
                    else:
                        job.update(status="COMPLETED",result=result["final"],routing_log=result["routing_log"],
                                   contributions=result["contributions"],usages=result["usages"])
                        job["cost_usd"]=sum(usage_cost(x) for x in result["usages"])
                        # No duplicar el costo de los spans LLM hijos en Phoenix.
                        span.set_attribute("task.cost_usd",job["cost_usd"])
                    span_output(span,{"status":job["status"],"result":job.get("result")})
            except asyncio.CancelledError:
                raise # Dejar RUNNING y processing para recuperacion; no inventar FAILED.
            except Exception as exc:
                # No exponer detalles del SDK que pueden contener datos de peticion.
                job.update(status="FAILED",error={"type":type(exc).__name__,
                           "message":"La ejecucion fallo; revisar la traza local"})
                log.error("job=%s failed type=%s",job_id,type(exc).__name__)
            job["finished_at"]=time.time()
            job["latency_seconds"]=job["finished_at"]-job["created_at"]
            await self.store.save(job)
            await self.store.redis.lrem(PROCESSING,0,job_id)

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        await self.store.redis.eval(
            "if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end",
            1,self.lease,self.owner)
