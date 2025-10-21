# SmartRAG Configuration Management

## 📋 Overview

SmartRAG uses a **single source of truth** configuration system with:

- ✅ **Pydantic validation** - Type-safe with human-readable error messages
- ✅ **Priority chain** - CLI flags > Env vars > YAML file > Defaults
- ✅ **Domain separation** - Organized into logical sections
- ✅ **Fail-fast validation** - Catches errors before runtime
- ✅ **Cross-validation** - Ensures consistency across domains

---

## 🎯 Configuration Priority

Configuration values are loaded in the following order (highest to lowest priority):

```
1. CLI Arguments / Explicit Overrides  ← Highest Priority
2. Environment Variables
3. config.yaml File
4. Hardcoded Defaults                  ← Lowest Priority
```

**Example**: If `config.yaml` sets `llm_model: "llama3.1:8b"` but you set environment variable `SMARTRAG_LLM_MODEL=llama2:7b`, the environment variable wins.

---

## 📁 Configuration Domains

Configuration is split into logical domains:

### 1. **system** - System-level settings

```yaml
system:
  name: "SmartRAG System"
  version: "2.0.0"
  offline_mode: true
  debug: false
  log_level: "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 2. **models** - AI/ML model configuration

```yaml
models:
  llm_type: "ollama" # ollama, openai, huggingface
  llm_model: "llama3.1:8b"
  ollama_host: "http://localhost:11434"
  embedding_model: "nomic-embed-text"
  embedding_dimension: 768 # Must match model output
  vision_model: "Salesforce/blip-image-captioning-base"
  whisper_model: "base" # tiny, base, small, medium, large
```

### 3. **vector_store** - Vector database settings

```yaml
vector_store:
  type: "chromadb" # chromadb, faiss
  persist_directory: "./vector_db"
  collection_name: "multimodal_documents"
  embedding_dimension: 768 # Must match models.embedding_dimension
  ollama_host: "http://localhost:11434" # Must match models.ollama_host
```

### 4. **processing** - Document/media processing

```yaml
processing:
  chunk_size: 1000 # 100-10000
  chunk_overlap: 200 # Must be < chunk_size
  max_image_size: [1024, 1024] # [width, height]
  ocr_enabled: true
  store_original_images: true
  image_preprocessing: "resize" # resize, crop, none
  audio_sample_rate: 16000 # Hz
  max_audio_duration: 300 # seconds
  batch_size: 32
```

### 5. **retrieval** - Search/retrieval settings

```yaml
retrieval:
  top_k: 5 # 1-50
  similarity_threshold: 0.7 # 0.0-1.0
  rerank_enabled: false
```

### 6. **generation** - LLM generation parameters

```yaml
generation:
  max_tokens: 2048 # 1-8192
  temperature: 0.7 # 0.0-2.0
  top_p: 0.9 # 0.0-1.0
  top_k: 50 # 1-100
  do_sample: true
  max_new_tokens: 1024
```

### 7. **ui** - User interface settings

```yaml
ui:
  title: "SmartRAG - Multimodal AI Assistant"
  page_icon: "🤖"
  layout: "wide" # wide, centered
  theme: "dark"
  show_recent_uploads: true
  max_upload_size_mb: 200
```

### 8. **storage** - File storage paths

```yaml
storage:
  data_directory: "./data"
  logs_directory: "./logs"
  cache_directory: "./cache"
  temp_uploads_directory: "./temp_uploads"
```

### 9. **supported_formats** - File type whitelist

```yaml
supported_formats:
  documents: [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"]
  images: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
  audio: [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
```

---

## 🚀 Usage Examples

### Basic Usage

```python
from config_schema import load_config

# Load from config.yaml with defaults
config = load_config()

# Access type-safe fields
print(config.models.llm_model)        # "llama3.1:8b"
print(config.generation.temperature)  # 0.7
print(config.retrieval.top_k)         # 5
```

### Override Parameters

```python
# Override using double underscore notation
config = load_config(
    "config.yaml",
    models__llm_model="llama2:7b",
    generation__temperature=0.5,
    retrieval__top_k=10
)
```

### Environment Variables

Set environment variables with `SMARTRAG_` prefix:

```bash
export SMARTRAG_LLM_MODEL=llama2:7b
export SMARTRAG_TEMPERATURE=0.5
export SMARTRAG_TOP_K=10
export SMARTRAG_DEBUG=true
```

These automatically override config.yaml values!

### Explicit Override Dictionary

```python
from config_schema import ConfigLoader

overrides = {
    'models': {
        'llm_model': 'llama3.1:70b',
        'ollama_host': 'http://remote-server:11434'
    },
    'vector_store': {
        'type': 'faiss'
    }
}

config = ConfigLoader.load(
    config_path="config.yaml",
    override_params=overrides
)
```

---

## ✅ Validation

The configuration system validates all values and provides helpful errors:

### Example Validation Errors

```python
# ❌ Invalid: chunk_overlap >= chunk_size
config = SmartRAGConfig(
    processing=ProcessingConfig(
        chunk_size=1000,
        chunk_overlap=1500  # ERROR!
    )
)
# ValueError: chunk_overlap (1500) must be less than chunk_size (1000)

# ❌ Invalid: temperature out of range
config = SmartRAGConfig(
    generation=GenerationConfig(temperature=3.0)  # ERROR!
)
# ValueError: temperature must be in [0.0, 2.0], got 3.0

# ❌ Invalid: embedding dimension mismatch
config = SmartRAGConfig(
    models=ModelsConfig(embedding_dimension=768),
    vector_store=VectorStoreConfig(embedding_dimension=384)  # WARNING!
)
# Automatically corrected with warning logged
```

---

## 🔧 Cross-Domain Validation

The system ensures consistency across configuration domains:

### Automatic Consistency Checks

1. **Embedding Dimension**: `models.embedding_dimension` must match `vector_store.embedding_dimension`
2. **Ollama Host**: `models.ollama_host` must match `vector_store.ollama_host`
3. **Chunk Overlap**: Must be less than `chunk_size`

If mismatches are detected, the system will:

- Log a warning
- Use the most appropriate value (usually from `models`)
- Continue execution with corrected config

---

## 📝 Best Practices

### 1. Use config.yaml for Defaults

Store your standard configuration in `config.yaml`:

```yaml
# config.yaml - Production defaults
models:
  llm_model: "llama3.1:8b"
  temperature: 0.7
```

### 2. Use Environment Variables for Deployment

Override settings per environment:

```bash
# Development
export SMARTRAG_DEBUG=true
export SMARTRAG_LOG_LEVEL=DEBUG

# Production
export SMARTRAG_DEBUG=false
export SMARTRAG_LOG_LEVEL=INFO
export SMARTRAG_OLLAMA_HOST=http://prod-server:11434
```

### 3. Use CLI Overrides for Testing

Quick one-off changes:

```python
# Test with different model
config = load_config(
    models__llm_model="llama2:7b",
    generation__temperature=0.3
)
```

### 4. Validate Early

Load configuration at startup to catch errors early:

```python
def main():
    try:
        config = load_config("config.yaml")
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    # Continue with validated config
    system = MultimodalRAGSystem(config_path="config.yaml")
```

### 5. Use Type Hints

The config objects are fully typed:

```python
def initialize_llm(config: SmartRAGConfig) -> OllamaLLM:
    # IDE will provide autocomplete and type checking
    model = config.models.llm_model
    temperature = config.generation.temperature
    return OllamaLLM(model, temperature)
```

---

## 🔍 Advanced Usage

### Multiple Environments

```python
def load_environment_config(env: str) -> SmartRAGConfig:
    """Load config for specific environment"""
    config_files = {
        'dev': 'config.dev.yaml',
        'staging': 'config.staging.yaml',
        'prod': 'config.yaml'
    }

    return load_config(
        config_files[env],
        system__debug=(env == 'dev'),
        system__log_level='DEBUG' if env == 'dev' else 'INFO'
    )

# Usage
config = load_environment_config('prod')
```

### Save Configuration

```python
from config_schema import save_config

config = load_config("config.yaml", models__llm_model="llama2:7b")
save_config(config, "config_modified.yaml")
```

### Convert to Dictionary

```python
config = load_config("config.yaml")
config_dict = config.to_dict()

# Use with legacy code that expects dict
legacy_system = OldSystem(config_dict)
```

---

## 🐛 Troubleshooting

### Issue: Configuration not loading

**Problem**: `config.yaml` not found

**Solution**:

```python
# Specify explicit path
config = load_config("path/to/config.yaml")

# Or use Path object
from pathlib import Path
config = load_config(Path(__file__).parent / "config.yaml")
```

### Issue: Environment variables not working

**Problem**: Env vars not overriding config.yaml

**Solution**: Ensure correct prefix and naming:

```bash
# ✅ Correct
export SMARTRAG_LLM_MODEL=llama2:7b

# ❌ Wrong - missing prefix
export LLM_MODEL=llama2:7b

# ❌ Wrong - incorrect casing (must be uppercase)
export smartrag_llm_model=llama2:7b
```

### Issue: Validation errors

**Problem**: Configuration values rejected

**Solution**: Check validation rules in error message:

```python
try:
    config = load_config()
except ValueError as e:
    print(f"Configuration error: {e}")
    # Error will specify which field and why it failed
```

### Issue: Type errors

**Problem**: IDE shows type errors

**Solution**: Use typed access:

```python
# ✅ Correct - fully typed
temperature: float = config.generation.temperature

# ❌ Avoid - loses type information
temperature = config.to_dict()['generation']['temperature']
```

---

## 📚 API Reference

### Main Functions

```python
def load_config(
    config_path: Optional[str] = None,
    **overrides
) -> SmartRAGConfig:
    """
    Load configuration with priority chain.

    Args:
        config_path: Path to YAML config file (default: "config.yaml")
        **overrides: Runtime overrides using double underscore notation
                    (e.g., models__llm_model="llama2:7b")

    Returns:
        SmartRAGConfig: Validated configuration object

    Raises:
        ValueError: If configuration is invalid
    """
```

```python
def save_config(
    config: SmartRAGConfig,
    output_path: Union[str, Path]
):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        output_path: Output file path
    """
```

### ConfigLoader Class

```python
class ConfigLoader:
    """Single source of truth configuration loader"""

    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        override_params: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> SmartRAGConfig:
        """Load configuration with full control"""
```

---

## 📖 Examples

See `config_examples.py` for comprehensive usage examples including:

- Basic loading
- Runtime overrides
- Environment variables
- Validation examples
- Production patterns
- Type-safe access

Run examples:

```bash
python config_examples.py
```

---

## 🔗 Related Files

- `config_schema.py` - Configuration schema and validation
- `config.yaml` - Default configuration file
- `.env.example` - Environment variable template
- `config_examples.py` - Usage examples
- `requirements.txt` - Includes `pydantic>=2.0.0`

---

## 💡 Migration Guide

### From Legacy Dict Config

**Before** (legacy):

```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

llm_model = config['models']['llm_model']  # No type checking
temperature = config.get('generation', {}).get('temperature', 0.7)  # Manual defaults
```

**After** (new system):

```python
from config_schema import load_config

config = load_config("config.yaml")  # Validated on load

llm_model = config.models.llm_model      # Type-safe, IDE autocomplete
temperature = config.generation.temperature  # Default already applied
```

### Benefits of Migration

✅ **Type Safety**: Catch errors at load time, not runtime  
✅ **IDE Support**: Full autocomplete and type hints  
✅ **Validation**: Automatic validation with clear error messages  
✅ **Consistency**: Cross-domain validation prevents mismatches  
✅ **Flexibility**: Easy overrides without modifying files  
✅ **Documentation**: Self-documenting with Pydantic models

---

## 📄 License

MIT License - See LICENSE file for details
