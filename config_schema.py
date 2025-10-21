"""
Configuration Schema and Management Pydantic validation
"""
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMType(str, Enum):
    """Supported LLM types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class VectorStoreType(str, Enum):
    """Supported vector store types"""
    CHROMADB = "chromadb"
    FAISS = "faiss"


class ImagePreprocessing(str, Enum):
    """Image preprocessing strategies"""
    RESIZE = "resize"
    CROP = "crop"
    NONE = "none"


# ============================================================================
# Configuration Domain Models
# ============================================================================

class SystemConfig(BaseModel):
    """System-level configuration"""
    name: str = Field(default="SmartRAG System", description="System name")
    version: str = Field(default="2.0.0", description="System version")
    offline_mode: bool = Field(default=True, description="Run in offline mode")
    debug: bool = Field(default=False, description="Enable debug logging")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{v}'")
        return v_upper


class ModelsConfig(BaseModel):
    """AI Models configuration"""
    # LLM Configuration
    llm_type: LLMType = Field(default=LLMType.OLLAMA, description="Type of LLM to use")
    llm_model: str = Field(default="llama3.1:8b", description="LLM model name")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    
    # Embedding Configuration
    embedding_model: str = Field(default="nomic-embed-text", description="Text embedding model")
    embedding_dimension: int = Field(default=768, description="Embedding vector dimension")
    
    # Vision Configuration
    vision_model: str = Field(
        default="Salesforce/blip-image-captioning-base",
        description="Vision/image captioning model"
    )
    
    # Speech Configuration
    whisper_model: str = Field(default="base", description="Whisper model size for audio")
    
    @field_validator('embedding_dimension')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"embedding_dimension must be positive, got {v}")
        if v > 4096:
            logger.warning(f"embedding_dimension {v} is very large, may impact performance")
        return v
    
    @field_validator('ollama_host')
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"ollama_host must start with http:// or https://, got '{v}'")
        return v.rstrip('/')  # Remove trailing slash


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    type: VectorStoreType = Field(default=VectorStoreType.CHROMADB, description="Vector store type")
    persist_directory: Path = Field(default=Path("./vector_db"), description="Storage directory")
    collection_name: str = Field(
        default="multimodal_documents",
        description="Collection/index name"
    )
    embedding_dimension: int = Field(default=768, description="Embedding dimension")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama host for embeddings")
    
    @field_validator('persist_directory')
    @classmethod
    def validate_persist_dir(cls, v: Path) -> Path:
        # Ensure path is absolute or resolve it
        if not v.is_absolute():
            v = Path.cwd() / v
        return v
    
    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("collection_name cannot be empty")
        # Sanitize collection name (no special chars for some vector stores)
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"collection_name must be alphanumeric with _/-, got '{v}'")
        return v.strip()


class ProcessingConfig(BaseModel):
    """Document and media processing configuration"""
    # Text chunking
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Chunk overlap size")
    
    # Image processing
    max_image_size: List[int] = Field(default=[1024, 1024], description="Max image dimensions [width, height]")
    ocr_enabled: bool = Field(default=True, description="Enable Tesseract OCR")
    store_original_images: bool = Field(default=True, description="Store original image data")
    image_preprocessing: ImagePreprocessing = Field(default=ImagePreprocessing.RESIZE, description="Image preprocessing")
    
    # Audio processing
    audio_sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate (Hz)")
    max_audio_duration: int = Field(default=300, ge=1, le=3600, description="Max audio length (seconds)")
    
    # Batch processing
    batch_size: int = Field(default=32, ge=1, le=128, description="Processing batch size")
    
    @field_validator('max_image_size')
    @classmethod
    def validate_image_size(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError(f"max_image_size must be [width, height], got {v}")
        if any(dim <= 0 or dim > 4096 for dim in v):
            raise ValueError(f"max_image_size dimensions must be in range [1, 4096], got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_chunk_overlap(self) -> 'ProcessingConfig':
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class RetrievalConfig(BaseModel):
    """Retrieval and search configuration"""
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    rerank_enabled: bool = Field(default=False, description="Enable result reranking")
    
    @field_validator('similarity_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"similarity_threshold must be in [0.0, 1.0], got {v}")
        return v


class GenerationConfig(BaseModel):
    """LLM generation configuration"""
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Max output tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    do_sample: bool = Field(default=True, description="Enable sampling")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096, description="Max new tokens to generate")
    
    @model_validator(mode='after')
    def validate_sampling_params(self) -> 'GenerationConfig':
        if self.max_new_tokens > self.max_tokens:
            logger.warning(
                f"max_new_tokens ({self.max_new_tokens}) > max_tokens ({self.max_tokens}), "
                f"adjusting max_new_tokens to {self.max_tokens}"
            )
            self.max_new_tokens = self.max_tokens
        return self


class UIConfig(BaseModel):
    """User interface configuration"""
    title: str = Field(default="SmartRAG - Multimodal AI Assistant", description="App title")
    page_icon: str = Field(default="🤖", description="Page icon emoji")
    layout: str = Field(default="wide", description="Page layout")
    theme: str = Field(default="dark", description="UI theme")
    show_recent_uploads: bool = Field(default=True, description="Show recent uploads section")
    max_upload_size_mb: int = Field(default=200, ge=1, le=1000, description="Max file upload size (MB)")
    
    @field_validator('layout')
    @classmethod
    def validate_layout(cls, v: str) -> str:
        valid_layouts = ['wide', 'centered']
        if v not in valid_layouts:
            raise ValueError(f"layout must be one of {valid_layouts}, got '{v}'")
        return v


class StorageConfig(BaseModel):
    """Storage and persistence configuration"""
    data_directory: Path = Field(default=Path("./data"), description="Data storage directory")
    logs_directory: Path = Field(default=Path("./logs"), description="Logs directory")
    cache_directory: Path = Field(default=Path("./cache"), description="Cache directory")
    temp_uploads_directory: Path = Field(default=Path("./temp_uploads"), description="Temporary uploads")
    
    @field_validator('data_directory', 'logs_directory', 'cache_directory', 'temp_uploads_directory')
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.is_absolute():
            v = Path.cwd() / v
        return v


class SupportedFormatsConfig(BaseModel):
    """Supported file formats"""
    documents: List[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"],
        description="Supported document formats"
    )
    images: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        description="Supported image formats"
    )
    audio: List[str] = Field(
        default=[".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        description="Supported audio formats"
    )
    
    @field_validator('documents', 'images', 'audio')
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        # Ensure all extensions start with a dot and are lowercase
        validated = []
        for ext in v:
            ext = ext.strip().lower()
            if not ext.startswith('.'):
                ext = f'.{ext}'
            validated.append(ext)
        return validated


# ============================================================================
# Main Configuration Model
# ============================================================================

class SmartRAGConfig(BaseModel):
    """Complete SmartRAG configuration with all domains"""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    supported_formats: SupportedFormatsConfig = Field(default_factory=SupportedFormatsConfig)
    
    @model_validator(mode='after')
    def validate_cross_domain(self) -> 'SmartRAGConfig':
        """Cross-domain validation"""
        # Ensure embedding dimensions match across configs
        if self.models.embedding_dimension != self.vector_store.embedding_dimension:
            logger.warning(
                f"Embedding dimension mismatch: models={self.models.embedding_dimension}, "
                f"vector_store={self.vector_store.embedding_dimension}. "
                f"Using models.embedding_dimension={self.models.embedding_dimension}"
            )
            self.vector_store.embedding_dimension = self.models.embedding_dimension
        
        # Ensure ollama_host is consistent
        if self.models.ollama_host != self.vector_store.ollama_host:
            logger.warning(
                f"Ollama host mismatch: models={self.models.ollama_host}, "
                f"vector_store={self.vector_store.ollama_host}. "
                f"Using models.ollama_host={self.models.ollama_host}"
            )
            self.vector_store.ollama_host = self.models.ollama_host
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)"""
        return {
            'system': self.system.model_dump(),
            'models': self.models.model_dump(),
            'vector_store': {
                **self.vector_store.model_dump(),
                'persist_directory': str(self.vector_store.persist_directory)
            },
            'processing': self.processing.model_dump(),
            'retrieval': self.retrieval.model_dump(),
            'generation': self.generation.model_dump(),
            'ui': self.ui.model_dump(),
            'storage': {
                k: str(v) for k, v in self.storage.model_dump().items()
            },
            'supported_formats': self.supported_formats.model_dump()
        }


# ============================================================================
# Configuration Loader (Single Source of Truth)
# ============================================================================

class ConfigLoader:
    
    DEFAULT_CONFIG_PATH = Path("config.yaml")
    ENV_PREFIX = "SMARTRAG_"
    
    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        override_params: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> SmartRAGConfig:
        """
        Load configuration with priority chain.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary with config values
            override_params: Explicit overrides (highest priority)
            validate: Whether to validate with Pydantic schema
            
        Returns:
            SmartRAGConfig: Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Step 1: Start with defaults
            config_data = cls._get_defaults()
            
            # Step 2: Load from YAML file
            yaml_config = cls._load_from_yaml(config_path)
            config_data = cls._deep_merge(config_data, yaml_config)
            
            # Step 3: Override with config_dict if provided
            if config_dict:
                config_data = cls._deep_merge(config_data, config_dict)
            
            # Step 4: Override with environment variables
            env_overrides = cls._load_from_env()
            config_data = cls._deep_merge(config_data, env_overrides)
            
            # Step 5: Apply explicit overrides (highest priority)
            if override_params:
                config_data = cls._deep_merge(config_data, override_params)
            
            # Step 6: Validate and return
            if validate:
                config = SmartRAGConfig(**config_data)
                logger.info("✅ Configuration loaded and validated successfully")
                return config
            else:
                # Return unvalidated dict (not recommended)
                logger.warning("⚠️  Configuration loaded without validation")
                return config_data
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            raise ValueError(f"Failed to load configuration: {e}") from e
    
    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get hardcoded defaults"""
        return SmartRAGConfig().to_dict()
    
    @classmethod
    def _load_from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.info(f"Config file not found: {config_path}, using defaults")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            logger.info(f"📄 Loaded configuration from {config_path}")
            return yaml_data or {}
        except Exception as e:
            logger.warning(f"Failed to load YAML from {config_path}: {e}")
            return {}
    
    @classmethod
    def _load_from_env(cls) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        env_config = {}
        
        # Map environment variables to config structure
        # Format: SMARTRAG_MODELS_LLM_MODEL=llama3.1:8b -> models.llm_model
        env_mappings = {
            f"{cls.ENV_PREFIX}DEBUG": "system.debug",
            f"{cls.ENV_PREFIX}LOG_LEVEL": "system.log_level",
            f"{cls.ENV_PREFIX}LLM_TYPE": "models.llm_type",
            f"{cls.ENV_PREFIX}LLM_MODEL": "models.llm_model",
            f"{cls.ENV_PREFIX}OLLAMA_HOST": "models.ollama_host",
            f"{cls.ENV_PREFIX}EMBEDDING_MODEL": "models.embedding_model",
            f"{cls.ENV_PREFIX}VISION_MODEL": "models.vision_model",
            f"{cls.ENV_PREFIX}VECTOR_STORE_TYPE": "vector_store.type",
            f"{cls.ENV_PREFIX}PERSIST_DIR": "vector_store.persist_directory",
            f"{cls.ENV_PREFIX}COLLECTION_NAME": "vector_store.collection_name",
            f"{cls.ENV_PREFIX}CHUNK_SIZE": "processing.chunk_size",
            f"{cls.ENV_PREFIX}TOP_K": "retrieval.top_k",
            f"{cls.ENV_PREFIX}TEMPERATURE": "generation.temperature",
            f"{cls.ENV_PREFIX}MAX_TOKENS": "generation.max_tokens",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert value to appropriate type
                value = cls._convert_env_value(value)
                cls._set_nested_value(env_config, config_path, value)
                logger.debug(f"🌍 Environment override: {env_var} -> {config_path} = {value}")
        
        return env_config
    
    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert string environment variable to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # String
        return value
    
    @staticmethod
    def _set_nested_value(data: Dict, path: str, value: Any):
        """Set nested dictionary value using dot notation path"""
        keys = path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries, override takes precedence"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides
) -> SmartRAGConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML config file
        **overrides: Explicit parameter overrides
        
    Returns:
        SmartRAGConfig: Validated configuration
        
    Example:
        >>> config = load_config("config.yaml", models__llm_model="llama3.1:8b")
    """
    # Convert double underscore kwargs to nested dict
    override_dict = {}
    for key, value in overrides.items():
        path = key.replace('__', '.')
        ConfigLoader._set_nested_value(override_dict, path, value)
    
    return ConfigLoader.load(
        config_path=config_path,
        override_params=override_dict
    )


def save_config(config: SmartRAGConfig, output_path: Union[str, Path]):
    """Save configuration to YAML file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"💾 Configuration saved to {output_path}")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("SmartRAG Configuration System - Testing")
    print("=" * 80)
    
    # Test 1: Load with defaults
    print("\n1. Loading with defaults...")
    config = load_config()
    print(f"✓ System: {config.system.name} v{config.system.version}")
    print(f"✓ LLM: {config.models.llm_type.value} - {config.models.llm_model}")
    print(f"✓ Vector Store: {config.vector_store.type.value}")
    
    # Test 2: Load from file
    print("\n2. Loading from config.yaml...")
    try:
        config = load_config("config.yaml")
        print(f"✓ Loaded from file successfully")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Test overrides
    print("\n3. Testing overrides...")
    config = load_config(
        models__llm_model="llama2:7b",
        generation__temperature=0.5,
        retrieval__top_k=10
    )
    print(f"✓ LLM Model: {config.models.llm_model}")
    print(f"✓ Temperature: {config.generation.temperature}")
    print(f"✓ Top K: {config.retrieval.top_k}")
    
    # Test 4: Validation
    print("\n4. Testing validation...")
    try:
        bad_config = SmartRAGConfig(
            processing=ProcessingConfig(chunk_overlap=1500, chunk_size=1000)
        )
        print("✗ Should have failed validation!")
    except Exception as e:
        print(f"✓ Validation works: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
