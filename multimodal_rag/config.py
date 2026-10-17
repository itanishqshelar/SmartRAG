"""
Configuration for SmartRAG.

Values are resolved in priority order:
  1. Environment variables, e.g. SMARTRAG_MODELS__LLM_MODEL=llama3.2:3b
  2. config.yaml (path overridable with SMARTRAG_CONFIG)
  3. Defaults defined below

Relative paths are resolved against the project root, not the working directory.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("SMARTRAG_CONFIG", PROJECT_ROOT / "config.yaml"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ModelsConfig(BaseModel):
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    vision_model: str = "Salesforce/blip-image-captioning-base"
    whisper_model: str = "base"

    @field_validator("ollama_host")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"ollama_host must start with http:// or https://, got '{v}'")
        return v.rstrip("/")


class VectorStoreConfig(BaseModel):
    persist_directory: Path = Path("./vector_db")
    collection_name: str = "multimodal_documents"

    @field_validator("persist_directory")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _resolve(v)


class ProcessingConfig(BaseModel):
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    max_image_size: Tuple[int, int] = (1024, 1024)
    ocr_enabled: bool = True
    max_audio_duration: int = Field(300, ge=1, le=7200, description="seconds")

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> "ProcessingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class RetrievalConfig(BaseModel):
    top_k: int = Field(5, ge=1, le=50)
    similarity_threshold: float = Field(0.45, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=16384)
    context_window: int = Field(8192, ge=512, le=131072)
    history_messages: int = Field(6, ge=0, le=50, description="Prior chat messages sent to the LLM")


class StorageConfig(BaseModel):
    data_directory: Path = Path("./data")

    @field_validator("data_directory")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _resolve(v)


class APIConfig(BaseModel):
    max_upload_size_mb: int = Field(50, ge=1, le=2000)
    cors_origins: List[str] = ["http://localhost:3000"]


class SmartRAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SMARTRAG_",
        env_nested_delimiter="__",
        yaml_file=CONFIG_PATH,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    models: ModelsConfig = ModelsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    processing: ProcessingConfig = ProcessingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, YamlConfigSettingsSource(settings_cls)


@lru_cache
def load_config() -> SmartRAGConfig:
    return SmartRAGConfig()
