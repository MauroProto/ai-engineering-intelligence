"""Fuentes recuperadas y respuesta citada."""

from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    text: str
    distance: float = Field(ge=0)


class RAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
