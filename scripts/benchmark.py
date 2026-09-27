"""Cinco peticiones simultaneas reales; resultados auditables sin claves."""
import asyncio
import json
import time
import sys
from datetime import datetime,timezone
from pathlib import Path
import httpx
import numpy as np

QUERIES=[
    "Como evita el supervisor un ciclo infinito y valida los aportes?",
    "Como se conserva un trabajo y su checkpoint cuando reinicia FastAPI?",
    "Que sucede si una persona rechaza una accion critica?",
    "Como se calculan p95 y costo sin inventar tokens?",
    "Cual fue la facturacion anual de la empresa en 2025?",
]


async def wait_job(client,job_id,target=None):
    deadline=time.monotonic()+650
    while time.monotonic()<deadline:
        response=await client.get(f"/tasks/{job_id}")
        response.raise_for_status()
        job=response.json()
        if (target and job["status"]==target) or job["status"] in ("COMPLETED","FAILED","CANCELLED"):
            return job
        await asyncio.sleep(.2)
    raise TimeoutError("El trabajo no alcanzo un estado esperado")


async def run():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8091",timeout=30) as client:
        health=await client.get("/health")
        health.raise_for_status()
        assert health.json()["model_provider"]=="deepseek"
        assert health.json()["credential_stored_in_files"] is False
        started=time.perf_counter()
        responses=await asyncio.gather(*[client.post("/tasks",json={"query":q}) for q in QUERIES])
        queue_latency=time.perf_counter()-started
        for response in responses:
            assert response.status_code==202,response.text
        ids=[r.json()["job_id"] for r in responses]
        jobs=await asyncio.gather(*[wait_job(client,i) for i in ids])
        assert all(j["status"]=="COMPLETED" for j in jobs),[(j["status"],j["error"]) for j in jobs]
        assert all(j["result"]["validation_passed"] for j in jobs)
        assert not jobs[-1]["result"]["answerable"],"La pregunta trampa no debe inventar datos"
        assert all(j["usages"] and sum(u["input_tokens"] for u in j["usages"])>0 for j in jobs)
        assert len(set(j["trace_id"] for j in jobs))==5
        latencies=[j["latency_seconds"] for j in jobs]
        report={"executed_at":datetime.now(timezone.utc).isoformat(),"mode":"live_deepseek_onnx_redis_phoenix",
                "model":health.json()["model"], "credential_stored_in_files":False,
                "samples":5,"concurrent_submission_seconds":queue_latency,
                "p95_seconds":float(np.percentile(latencies,95)),
                "cost_usd":sum(j["cost_usd"] for j in jobs),"jobs":jobs,
                "caveat":"Cinco muestras no definen un SLO. Tokens LLM medidos por DeepSeek; embeddings sin contador. Costo estimado con tarifas OFF-PEAK verificadas, no factura."}
        Path("evidence").mkdir(exist_ok=True)
        Path("evidence/benchmark.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print({k:v for k,v in report.items() if k!="jobs"})
        if "--include-hitl" not in sys.argv:
            print("Corrida de cinco consultas guardada. HITL se verifica separadamente con --include-hitl.")
            return
        # HITL positivo y negativo, y rechazo de doble aprobacion.
        hitl=[]
        for approved in (True,False):
            created=await client.post("/tasks",json={"query":"Explica la politica de aprobacion humana","critical_action":True})
            job_id=created.json()["job_id"]
            waiting=await wait_job(client,job_id,"WAITING_APPROVAL")
            assert waiting["status"]=="WAITING_APPROVAL"
            response=await client.post(f"/tasks/{job_id}/approve",json={"approved":approved,"reason":"Decision de prueba local"})
            assert response.status_code==202,response.text
            duplicate=await client.post(f"/tasks/{job_id}/approve",json={"approved":approved,"reason":"Repetida"})
            assert duplicate.status_code==409
            result=await wait_job(client,job_id)
            assert result["status"]==("COMPLETED" if approved else "CANCELLED"),result
            hitl.append(result)
        invalid=await client.post("/tasks",json={"query":"x"})
        assert invalid.status_code==422
        Path("evidence/hitl.json").write_text(json.dumps(hitl,ensure_ascii=False,indent=2),encoding="utf-8")
        print("HITL aprobar/rechazar, doble aprobacion y contrato API: OK")


if __name__=="__main__":
    asyncio.run(run())
