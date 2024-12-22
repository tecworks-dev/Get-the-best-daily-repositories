# Quick Start Guide

Get started with LLM Karpathy Consortium in minutes.

## Installation

```bash
# Install package
pip install llm-karpathy-consortium

# Verify installation
python -c "import llm_consortium; print(llm_consortium.__version__)"
```

## Basic Usage

```python
from llm_consortium import ConsortiumOrchestrator

# Initialize orchestrator
orchestrator = ConsortiumOrchestrator(
    models=["claude-3-sonnet-20240229", "gpt-4"]
)

# Process a prompt
async def get_response():
    result = await orchestrator.orchestrate(
        "Explain quantum computing in simple terms"
    )
    print(result['synthesis']['synthesis'])

# Run the async function
import asyncio
asyncio.run(get_response())
```

## Configuration

```python
# Custom configuration
config = {
    "confidence_threshold": 0.8,
    "max_iterations": 3,
    "arbiter_model": "claude-3-opus-20240229"
}

orchestrator = ConsortiumOrchestrator(**config)
```

## Common Operations

### Multiple Models
```python
orchestrator = ConsortiumOrchestrator(
    models=[
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "gpt-4",
        "gemini-pro"
    ]
)
```

### Error Handling
```python
try:
    result = await orchestrator.orchestrate("Your prompt")
except Exception as e:
    print(f"Error: {e}")
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Next Steps

1. Read the [Architecture Overview](ARCHITECTURE.md)
2. Explore [Integration Patterns](INTEGRATION_PATTERNS.md)
3. Review [Security Guidelines](SECURITY.md)
4. Check [Examples](../examples/)

## Common Issues

1. Model Connection
```python
# Check model availability
from llm_consortium.health import check_model_health
status = await check_model_health()
```

2. Performance
```python
# Enable performance monitoring
from llm_consortium.metrics import MetricsCollector
metrics = MetricsCollector().collect()
```

## Support

- Create an issue on GitHub
- Join our Discord community
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)

## Best Practices

1. Always use async/await properly
2. Handle errors appropriately
3. Monitor performance metrics
4. Follow security guidelines
5. Keep dependencies updated

## Additional Resources

- [Full Documentation](INDEX.md)
- [API Reference](../API.md)
- [Examples](../examples/)
- [Community Forum](https://github.com/yourusername/llm-karpathy-consortium/discussions)
