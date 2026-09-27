"""Contratos HTTP simulados identificados; embeddings ONNX reales en test separado."""
import json
import httpx
import pytest
from pydantic import ValidationError
from app.config import Settings
from app.llm import ModelGateway, ModelUnavailableError, CachedEmbeddings
from app.observability import usage_cost
from app.state import RouteDecision


def test_settings_reject_other_endpoints_credentials_and_unapproved_models():
    for endpoint in ("https://api.openai.com", "http://api.deepseek.com",
                     "https://secret@api.deepseek.com", "https://api.deepseek.com/other"):
        with pytest.raises(ValidationError):
            Settings(deepseek_base_url=endpoint, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(llm_model="gpt-oss:120b-cloud", _env_file=None)


def test_local_billing_does_not_apply_cloud_prices():
    for provider in ("onnx_local",):
        assert usage_cost({"provider":provider,"input_tokens":1000,"output_tokens":500})==0


@pytest.mark.asyncio
async def test_chat_ignores_keys_and_proxy_env_and_counts_real_response_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-test-value")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "authorized-test-placeholder")
    monkeypatch.setenv("HTTP_PROXY", "http://unreachable.invalid:9999")
    observed=[]
    def handler(request):
        observed.append(request)
        assert request.url.host=="api.deepseek.com"
        assert request.headers["authorization"]=="Bearer authorized-test-placeholder"
        assert "x-api-key" not in request.headers
        if request.url.path=="/models":
            return httpx.Response(200,json={"data":[{"id":"deepseek-flash"}]})
        payload=json.loads(request.content)
        assert payload["response_format"]["type"]=="json_object"
        assert payload["stream"] is False and payload["thinking"]["type"]=="disabled"
        return httpx.Response(200,json={"model":"deepseek-flash",
            "choices":[{"finish_reason":"stop","message":{"content":'{"next_agent":"researcher","reason":"buscar fuentes"}'}}],
            "usage":{"prompt_tokens":54,"completion_tokens":18}})
    gateway=ModelGateway(transport=httpx.MockTransport(handler))
    try:
        result,usage=await gateway.structured(RouteDecision,"Router",{"query":"investigar"})
        assert result.next_agent=="researcher"
        assert usage["provider"]=="deepseek" and usage["tokens_measured"]
        assert usage["input_tokens"]==54 and usage["output_tokens"]==18
        assert usage["cost_usd"]>0
        assert len(observed)==2
        assert gateway.client._trust_env is False
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_missing_key_never_reads_other_provider_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY",raising=False)
    monkeypatch.setenv("OPENAI_API_KEY","unauthorized-test-placeholder")
    paths=[]
    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(404,json={"error":"model not found"})
    gateway=ModelGateway(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ModelUnavailableError):
            await gateway.structured(RouteDecision,"Router",{})
        assert paths==[]
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_redirect_is_not_followed():
    paths=[]
    def handler(request):
        paths.append(str(request.url))
        return httpx.Response(302,headers={"location":"https://unauthorized.invalid"})
    gateway=ModelGateway(transport=httpx.MockTransport(handler),api_key="test-placeholder")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await gateway.validate_environment()
        assert paths==["https://api.deepseek.com/models"]
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_retry_includes_tokens_from_invalid_json():
    count=0
    def handler(request):
        nonlocal count
        if request.url.path=="/models":
            return httpx.Response(200,json={"data":[{"id":"deepseek-flash"}]})
        count+=1
        content='{"invalid":true}' if count==1 else '{"next_agent":"researcher","reason":"ok"}'
        return httpx.Response(200,json={"model":"deepseek-flash",
            "choices":[{"finish_reason":"stop","message":{"content":content}}],
            "usage":{"prompt_tokens":12,"completion_tokens":8}})
    gateway=ModelGateway(transport=httpx.MockTransport(handler),api_key="test-placeholder")
    try:
        _,usage=await gateway.structured(RouteDecision,"Router",{})
        assert usage["attempts"]==2
        assert usage["input_tokens"]==24 and usage["output_tokens"]==16
    finally:
        await gateway.close()


def test_embedding_cache_missing_is_not_downloaded(tmp_path,monkeypatch):
    monkeypatch.setattr(CachedEmbeddings,"DOWNLOAD_PATH",tmp_path)
    with pytest.raises(ModelUnavailableError,match="ONNX"):
        CachedEmbeddings()._download_model_if_not_exists()
    assert list(tmp_path.iterdir())==[]


@pytest.mark.asyncio
async def test_real_cached_onnx_embeddings_without_any_network(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("No se permite ninguna descarga o llamada HTTP en embeddings")
    monkeypatch.setattr(httpx.Client,"send",forbidden)
    monkeypatch.setattr(httpx.AsyncClient,"send",forbidden)
    gateway=ModelGateway()
    try:
        vectors,usage=await gateway.embeddings(["Redis persiste checkpoints", "Un gato duerme"])
        assert len(vectors)==2 and len(vectors[0])==384
        assert vectors[0]!=vectors[1]
        assert usage["provider"]=="onnx_local" and not usage["tokens_measured"]
        assert usage["cost_usd"]==0
    finally:
        await gateway.close()
