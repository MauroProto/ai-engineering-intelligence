"""Indice Chroma y embeddings ONNX reales, sin un LLM ni servicios pagos."""
import pytest
from app.config import settings
from app.llm import ModelGateway
from app.rag import HybridKnowledgeBase


@pytest.mark.asyncio
async def test_real_local_index_is_idempotent_and_retrieves_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"data_dir",tmp_path)
    gateway=ModelGateway()
    try:
        knowledge=HybridKnowledgeBase(gateway)
        await knowledge.setup()
        initial_ids={chunk["id"] for chunk in knowledge.chunks}
        assert len(initial_ids)==4
        assert knowledge.collection.count()==4
        assert knowledge.ingestion_usage["provider"]=="onnx_local"
        await knowledge.setup()
        assert knowledge.collection.count()==4
        assert {chunk["id"] for chunk in knowledge.chunks}==initial_ids
        assert knowledge.ingestion_usage is None
        for question,source in (
            ("Redis checkpoints cola workers reinicio", "operacion.md"),
            ("Aprobacion humana interrupt rechazo approved false CANCELLED", "seguridad.md"),
            ("p95 percentil 95 latencia cinco muestras benchmark", "metricas.md"),
        ):
            rows,usage=await knowledge.search(question,limit=2)
            assert source in {row["source"] for row in rows},rows
            assert len({row["id"] for row in rows})==len(rows)
            assert {row["id"] for row in rows}.issubset(initial_ids)
            assert usage["cost_usd"]==0 and not usage["tokens_measured"]
    finally:
        await gateway.close()
