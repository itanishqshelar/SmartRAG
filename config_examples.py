"""
SmartRAG Configuration Examples
Demonstrates how to use the new configuration system
"""

from config_schema import (
    load_config, 
    SmartRAGConfig,
    ConfigLoader,
    save_config
)
from pathlib import Path

print("=" * 80)
print("SmartRAG Configuration System - Usage Examples")
print("=" * 80)

# ============================================================================
# Example 1: Load default configuration
# ============================================================================
print("\n📋 Example 1: Load with defaults")
print("-" * 80)

config = load_config()
print(f"✓ System: {config.system.name}")
print(f"✓ LLM: {config.models.llm_type.value} - {config.models.llm_model}")
print(f"✓ Embedding: {config.models.embedding_model} ({config.models.embedding_dimension}d)")
print(f"✓ Vector Store: {config.vector_store.type.value}")
print(f"✓ Chunk Size: {config.processing.chunk_size}")
print(f"✓ Top K: {config.retrieval.top_k}")


# ============================================================================
# Example 2: Load from config.yaml
# ============================================================================
print("\n📋 Example 2: Load from config.yaml")
print("-" * 80)

try:
    config = load_config("config.yaml")
    print(f"✓ Loaded from config.yaml")
    print(f"  System: {config.system.name} v{config.system.version}")
    print(f"  Collection: {config.vector_store.collection_name}")
except Exception as e:
    print(f"✗ Failed to load: {e}")


# ============================================================================
# Example 3: Override specific parameters
# ============================================================================
print("\n📋 Example 3: Runtime overrides with double underscore notation")
print("-" * 80)

config = load_config(
    "config.yaml",
    models__llm_model="llama2:7b",  # Override LLM model
    generation__temperature=0.5,     # Lower temperature
    retrieval__top_k=10,             # More results
    processing__chunk_size=500       # Smaller chunks
)

print(f"✓ LLM Model: {config.models.llm_model} (overridden)")
print(f"✓ Temperature: {config.generation.temperature} (overridden)")
print(f"✓ Top K: {config.retrieval.top_k} (overridden)")
print(f"✓ Chunk Size: {config.processing.chunk_size} (overridden)")


# ============================================================================
# Example 4: Direct ConfigLoader usage with explicit overrides
# ============================================================================
print("\n📋 Example 4: Explicit override dictionary")
print("-" * 80)

overrides = {
    'models': {
        'llm_model': 'llama3.1:70b',
        'ollama_host': 'http://remote-server:11434'
    },
    'vector_store': {
        'type': 'faiss',
        'collection_name': 'production_docs'
    }
}

config = ConfigLoader.load(
    config_path="config.yaml",
    override_params=overrides
)

print(f"✓ LLM Model: {config.models.llm_model}")
print(f"✓ Ollama Host: {config.models.ollama_host}")
print(f"✓ Vector Store: {config.vector_store.type.value}")
print(f"✓ Collection: {config.vector_store.collection_name}")


# ============================================================================
# Example 5: Environment variable overrides
# ============================================================================
print("\n📋 Example 5: Environment variable overrides")
print("-" * 80)
print("Set environment variables like:")
print("  export SMARTRAG_LLM_MODEL=llama2:7b")
print("  export SMARTRAG_TEMPERATURE=0.3")
print("  export SMARTRAG_TOP_K=20")
print("\nThese will automatically override config.yaml values!")
print("Priority: CLI args > Env vars > YAML file > Defaults")


# ============================================================================
# Example 6: Access typed configuration
# ============================================================================
print("\n📋 Example 6: Type-safe configuration access")
print("-" * 80)

config = load_config("config.yaml")

# All fields are type-checked and validated
print(f"✓ Temperature (float): {config.generation.temperature}")
print(f"✓ Top K (int): {config.retrieval.top_k}")
print(f"✓ OCR Enabled (bool): {config.processing.ocr_enabled}")
print(f"✓ Max Image Size (list): {config.processing.max_image_size}")
print(f"✓ LLM Type (enum): {config.models.llm_type}")


# ============================================================================
# Example 7: Validation errors
# ============================================================================
print("\n📋 Example 7: Configuration validation")
print("-" * 80)

try:
    # This should fail validation (chunk_overlap >= chunk_size)
    from config_schema import ProcessingConfig, SmartRAGConfig
    
    bad_config = SmartRAGConfig(
        processing=ProcessingConfig(
            chunk_size=1000,
            chunk_overlap=1500  # Invalid: overlap > size
        )
    )
except ValueError as e:
    print(f"✓ Validation caught error: {e}")


try:
    # This should fail validation (invalid temperature range)
    from config_schema import GenerationConfig
    
    bad_gen = GenerationConfig(temperature=3.0)  # Invalid: > 2.0
except ValueError as e:
    print(f"✓ Validation caught error: {e}")


# ============================================================================
# Example 8: Save configuration
# ============================================================================
print("\n📋 Example 8: Save configuration to file")
print("-" * 80)

config = load_config("config.yaml")
output_path = Path("config_backup.yaml")
save_config(config, output_path)
print(f"✓ Configuration saved to {output_path}")


# ============================================================================
# Example 9: Convert to dictionary for backward compatibility
# ============================================================================
print("\n📋 Example 9: Convert to dictionary")
print("-" * 80)

config = load_config("config.yaml")
config_dict = config.to_dict()

print(f"✓ Converted to dict with {len(config_dict)} top-level keys:")
print(f"  Keys: {', '.join(config_dict.keys())}")


# ============================================================================
# Example 10: Production usage pattern
# ============================================================================
print("\n📋 Example 10: Production initialization pattern")
print("-" * 80)

def initialize_smartrag(env: str = "production"):
    """Production-grade configuration loading"""
    
    # Different configs for different environments
    config_files = {
        'development': 'config.dev.yaml',
        'staging': 'config.staging.yaml',
        'production': 'config.yaml'
    }
    
    config_file = config_files.get(env, 'config.yaml')
    
    try:
        # Load with validation
        config = load_config(
            config_file,
            # Override debug mode based on environment
            system__debug=(env == 'development'),
            system__log_level='DEBUG' if env == 'development' else 'INFO'
        )
        
        print(f"✓ SmartRAG initialized for {env} environment")
        print(f"  Config: {config_file}")
        print(f"  Debug: {config.system.debug}")
        print(f"  Log Level: {config.system.log_level}")
        
        return config
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        raise

# Simulate production initialization
config = initialize_smartrag('production')


print("\n" + "=" * 80)
print("✅ All examples completed successfully!")
print("=" * 80)
