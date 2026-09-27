"""API desacoplada. El procesamiento simulado está solicitado en esta consigna.

No es el worker del proyecto final: esta cola simple no recupera trabajos que
queden en running tras una caída abrupta. El proyecto utiliza checkpoints.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from redis import asyncio as aioredis
from redis.exceptions import RedisError

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6389/1")
NAMESPACE = os.getenv("M7U2_NAMESPACE", "m7u2")
QUEUE_NAME = f"{NAMESPACE}:tasks_queue"
STATUS_PREFIX = f"{NAMESPACE}:status:"
logger = logging.getLogger(__name__)
redis_client = None


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000)


class TaskStatus(BaseModel):
    job_id: uuid.UUID
    status: Literal["pending", "running", "completed", "failed"]
    query: str
    result: str | None = None
    error: str | None = None


async def simulated_processing(query: str) -> str:
    """Simula latencia sin bloquear el event loop. No llama a un LLM."""
    await asyncio.sleep(0.1)
    return f"Procesamiento simulado completado para: {query}"


async def task_worker():
    while True:
        job = None
        try:
            item = await redis_client.blpop(QUEUE_NAME, timeout=1)
            if item is None:
                continue
            _, job_id = item
            raw = await redis_client.get(STATUS_PREFIX + job_id)
            if raw is None:
                logger.warning("Trabajo ausente o expirado; se omite")
                continue
            job = TaskStatus.model_validate_json(raw)
            job.status = "running"
            await redis_client.set(STATUS_PREFIX + job_id, job.model_dump_json(), ex=86400)
            async with asyncio.timeout(30):
                job.result = await simulated_processing(job.query)
            job.status = "completed"
            await redis_client.set(STATUS_PREFIX + job_id, job.model_dump_json(), ex=86400)
        except asyncio.CancelledError:
            if job is not None:
                job.status = "failed"
                job.error = "Worker detenido durante el procesamiento"
                await redis_client.set(STATUS_PREFIX + str(job.job_id), job.model_dump_json(), ex=86400)
            raise
        except Exception as exc:
            logger.error("Error controlado del worker: %s", type(exc).__name__)
            if job is not None:
                job.status = "failed"
                job.error = type(exc).__name__  # No divulgar detalles internos ni credenciales.
                await redis_client.set(STATUS_PREFIX + str(job.job_id), job.model_dump_json(), ex=86400)
            await asyncio.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    workers = [asyncio.create_task(task_worker()) for _ in range(3)]
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await redis_client.aclose()


app = FastAPI(title="Ejercicio M7 U2", lifespan=lifespan)


@app.post("/process", status_code=202, response_model=TaskStatus)
async def create_task(request: TaskRequest):
    job = TaskStatus(job_id=uuid.uuid4(), status="pending", query=request.query)
    try:
        # Una transacción hace visible el estado antes de que el worker vea la cola.
        async with redis_client.pipeline(transaction=True) as pipeline:
            pipeline.set(STATUS_PREFIX + str(job.job_id), job.model_dump_json(), ex=86400)
            pipeline.rpush(QUEUE_NAME, str(job.job_id))
            await pipeline.execute()
    except RedisError:
        raise HTTPException(status_code=503, detail="Cola temporalmente no disponible") from None
    return job


@app.get("/status/{job_id}", response_model=TaskStatus)
async def get_status(job_id: uuid.UUID):
    try:
        raw = await redis_client.get(STATUS_PREFIX + str(job_id))
    except RedisError:
        raise HTTPException(status_code=503, detail="Estado temporalmente no disponible") from None
    if raw is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado o expirado")
    return TaskStatus.model_validate_json(raw)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
