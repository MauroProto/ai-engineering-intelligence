"""Contrato estricto para la extracción técnica."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Criticality(StrEnum):
    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"


class TechnicalExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tecnologias: list[str] = Field(min_length=1)
    nivel_de_criticidad: Criticality
    resumen_tecnico: str = Field(min_length=20)
