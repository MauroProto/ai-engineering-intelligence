from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_model: Literal["deepseek-flash"] = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    embedding_model: Literal["all-MiniLM-L6-v2"] = "all-MiniLM-L6-v2"
    redis_url: str = "redis://127.0.0.1:6389/0"
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    phoenix_project_name: str = "coderhouse-intelligence"
    data_dir: Path = Path(".data")
    worker_concurrency: int = Field(default=5, ge=1, le=16)
    max_agent_steps: int = Field(default=8, ge=4, le=20)
    task_timeout_seconds: int = Field(default=600, ge=5, le=600)
    model_timeout_seconds: int = Field(default=120, ge=5, le=300)
    # DeepSeek Flash, tarifas OFF-PEAK verificadas 26/09/2026 (fin de semana).
    input_price_per_million: float = Field(default=.15, ge=0)
    cached_input_price_per_million: float = Field(default=.003, ge=0)
    output_price_per_million: float = Field(default=.60, ge=0)
    embedding_price_per_million: float = Field(default=.02, ge=0)

    @field_validator("deepseek_base_url")
    @classmethod
    def official_endpoint_only(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.hostname != "api.deepseek.com"
                or parsed.username or parsed.password or parsed.port not in (None,443)
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("La credencial solo puede enviarse al HTTPS oficial de DeepSeek")
        return value.rstrip("/")


settings = Settings()
