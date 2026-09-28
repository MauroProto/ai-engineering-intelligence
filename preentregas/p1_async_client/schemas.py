"""Contratos de entrada y salida para ambos proveedores."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str
    text: str = ""
    finish_reason: str | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def success_or_error(self) -> "ModelResponse":
        if bool(self.text) == bool(self.error_code):
            raise ValueError("La respuesta debe tener texto o un error controlado")
        return self
