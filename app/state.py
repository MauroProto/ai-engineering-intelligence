from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TaskRequest(Contract):
    query: str = Field(min_length=3, max_length=1500)
    critical_action: bool = False


class ApprovalRequest(Contract):
    approved: bool
    reason: str = Field(min_length=3, max_length=300)


class Evidence(Contract):
    id: str
    text: str
    source: str
    page: int = Field(ge=1)
    category: str
    score: float


class ResearchArtifact(Contract):
    summary: str
    source_ids: list[str]
    sufficient: bool


class AnalysisArtifact(Contract):
    answer: str
    source_ids: list[str]
    answerable: bool
    confidence: float = Field(ge=0, le=1)
    limitations: str


class RouteDecision(Contract):
    next_agent: Literal["researcher", "analyst", "validate"]
    reason: str


class FinalResponse(Contract):
    answer: str
    references: list[Evidence]
    answerable: bool
    confidence: float = Field(ge=0, le=1)
    limitations: str
    contributors: list[str]
    validation_passed: bool


class Contribution(Contract):
    agent: str
    artifact: dict
    tool: str


def merge_contributions(left: list, right: list) -> list:
    # No mutar parcelas del estado recibidas; cada nodo escribe su aporte.
    return [*left, *right]


class IntelligenceState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    critical_action: bool
    approved: bool
    declined: bool
    next_agent: str
    steps: int
    refinements: int
    evidence: list[dict]
    research: dict
    analysis: dict
    contributions: Annotated[list[dict], merge_contributions]
    validation_errors: list[str]
    validated: bool
    final: dict
    usages: Annotated[list[dict], merge_contributions]
    routing_log: Annotated[list[dict], merge_contributions]
