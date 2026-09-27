import json
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from .config import settings

provider = TracerProvider(resource=Resource.create({
    "openinference.project.name":settings.phoenix_project_name,
    "service.name":"coderhouse-intelligence",
}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
    endpoint=settings.phoenix_collector_endpoint),schedule_delay_millis=500))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("coderhouse.intelligence")


def span_input(span, value, kind="CHAIN"):
    span.set_attribute("openinference.span.kind", kind)
    span.set_attribute("input.value", json.dumps(value, ensure_ascii=False))
    span.set_attribute("input.mime_type", "application/json")


def span_output(span, value):
    span.set_attribute("output.value", json.dumps(value, ensure_ascii=False))
    span.set_attribute("output.mime_type", "application/json")


def usage_cost(usage):
    # Cero facturacion API no significa que hardware/energia sean gratuitos.
    if usage.get("provider") in ("ollama_local", "onnx_local"):
        return 0.0
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cached = min(inp, usage.get("cached_tokens", 0))
    if usage.get("kind") == "embedding":
        return inp * settings.embedding_price_per_million / 1_000_000
    return ((inp-cached)*settings.input_price_per_million
            + cached*settings.cached_input_price_per_million
            + out*settings.output_price_per_million) / 1_000_000
