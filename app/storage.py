import json
import time
from uuid import uuid4
from .state import TaskRequest,ApprovalRequest

PREFIX="intelligence:"
PENDING=PREFIX+"queue"
PROCESSING=PREFIX+"processing"


class JobStore:
    def __init__(self,redis):
        self.redis=redis

    async def create(self,request:TaskRequest):
        job_id=str(uuid4())
        job={"job_id":job_id,"status":"QUEUED","created_at":time.time(),
             "request":request.model_dump(),"error":None,"result":None}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(PREFIX+"job:"+job_id,json.dumps(job))
            pipe.lpush(PENDING,job_id)
            await pipe.execute()
        return job

    async def get(self,job_id):
        raw=await self.redis.get(PREFIX+"job:"+job_id)
        return json.loads(raw) if raw else None

    async def save(self,job):
        await self.redis.set(PREFIX+"job:"+job["job_id"],json.dumps(job,ensure_ascii=False))

    async def approve(self,job_id,decision:ApprovalRequest):
        # WATCH evita doble aprobacion y doble encolado.
        from redis.exceptions import WatchError
        key=PREFIX+"job:"+job_id
        for _ in range(3):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw=await pipe.get(key)
                    if not raw:
                        raise KeyError(job_id)
                    job=json.loads(raw)
                    if job["status"]!="WAITING_APPROVAL":
                        raise ValueError("El trabajo no esta esperando aprobacion")
                    job.update(status="QUEUED",approval=decision.model_dump(),resume_pending=True)
                    pipe.multi()
                    pipe.set(key,json.dumps(job))
                    pipe.lpush(PENDING,job_id)
                    await pipe.execute()
                    return job
                except WatchError:
                    continue
        raise ValueError("Aprobacion concurrente; consultar el estado actualizado")

    async def all_jobs(self):
        keys=[key async for key in self.redis.scan_iter(PREFIX+"job:*")]
        if not keys:
            return []
        return [json.loads(x) for x in await self.redis.mget(keys) if x]
