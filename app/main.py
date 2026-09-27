from contextlib import asynccontextmanager
from uuid import UUID
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from redis.asyncio import Redis
from .config import settings
from .state import TaskRequest,ApprovalRequest
from .storage import JobStore
from .llm import ModelGateway
from .rag import HybridKnowledgeBase
from .graph import build_graph
from .checkpointer import RedisCheckpointSaver
from .worker import WorkerPool


@asynccontextmanager
async def lifespan(app):
    redis=Redis.from_url(settings.redis_url,decode_responses=False)
    await redis.ping()
    gateway=ModelGateway()
    pool=None
    try:
        await gateway.validate_environment()
        knowledge=HybridKnowledgeBase(gateway)
        await knowledge.setup()
        store=JobStore(redis)
        graph=build_graph(gateway,knowledge,RedisCheckpointSaver(redis))
        pool=WorkerPool(store,graph)
        await pool.start()
        app.state.store,app.state.graph=store,graph
        yield
    finally:
        if pool is not None:
            await pool.stop()
        await gateway.close()
        await redis.aclose()


app=FastAPI(title="Sistema Intelligence",version="1.0.0",lifespan=lifespan,
            description="Laboratorio local de RAG multiagente con Redis, Phoenix y HITL")


@app.post("/tasks",status_code=202)
async def create_task(request:TaskRequest):
    job=await app.state.store.create(request)
    return {"job_id":job["job_id"],"status":job["status"]}


@app.get("/tasks/{job_id}")
async def task_status(job_id:UUID):
    job=await app.state.store.get(str(job_id))
    if not job:
        raise HTTPException(404,"Trabajo inexistente")
    return job


@app.post("/tasks/{job_id}/approve",status_code=202)
async def approve(job_id:UUID,decision:ApprovalRequest):
    try:
        job=await app.state.store.approve(str(job_id),decision)
        return {"job_id":job["job_id"],"status":job["status"]}
    except KeyError:
        raise HTTPException(404,"Trabajo inexistente") from None
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from None


@app.get("/health")
async def health():
    await app.state.store.redis.ping()
    return {"status":"ok","persistence":"redis","checkpoint":"RedisCheckpointSaver",
            "model_provider":"deepseek","model":settings.llm_model,
            "embedding_provider":"onnx_local","credential_stored_in_files":False}


@app.get("/metrics")
async def metrics():
    jobs=await app.state.store.all_jobs()
    completed=[job for job in jobs if job["status"]=="COMPLETED"]
    latencies=[x["latency_seconds"] for x in completed]
    return {"completed":len(completed),"failed":sum(j["status"]=="FAILED" for j in jobs),
            "p95_seconds":float(np.percentile(latencies,95)) if latencies else None,
            "total_cost_usd":sum(x.get("cost_usd",0) for x in completed),
            "jobs":[{k:j.get(k) for k in ("job_id","status","latency_seconds","cost_usd","trace_id")}
                    for j in sorted(jobs,key=lambda x:x["created_at"],reverse=True)[:30]],
            "note":"p95 de trabajos COMPLETED desde encolado; muestra pequena, no SLO de produccion"}


@app.get("/dashboard",response_class=HTMLResponse)
async def dashboard():
    # textContent, nunca innerHTML: no ejecutar texto de preguntas o de modelos.
    return '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Monitoreo Intelligence</title>
    <style>body{font:17px system-ui;max-width:1100px;margin:36px auto;color:#222}table{border-collapse:collapse;width:100%;font:13px monospace}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}a{color:#245}</style>
    <h1>Monitoreo del Sistema Intelligence</h1><p>Datos reales de Redis. DeepSeek Flash con tokens medidos. USD = estimacion API, no factura.</p>
    <p><a href="http://127.0.0.1:6006">Ver trazas en Phoenix</a> · <a href="/docs">Documentacion API</a></p>
    <pre id="summary"></pre><table><thead><tr><th>job_id</th><th>Estado</th><th>Latencia (s)</th><th>USD</th><th>trace_id</th></tr></thead><tbody id="rows"></tbody></table>
    <script>async function refresh(){const d=await(await fetch('/metrics')).json();document.getElementById('summary').textContent=JSON.stringify({...d,jobs:undefined},null,2);const body=document.getElementById('rows');body.replaceChildren();for(const job of d.jobs){const tr=document.createElement('tr');for(const key of ['job_id','status','latency_seconds','cost_usd','trace_id']){const td=document.createElement('td');td.textContent=job[key]??'-';tr.append(td)}body.append(tr)}}refresh();setInterval(refresh,3000)</script></html>'''
