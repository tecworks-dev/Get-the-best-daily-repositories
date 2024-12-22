# Documentation Style Guide

## General Principles

1. Clarity
   - Write in clear, simple language
   - Avoid jargon unless necessary
   - Define technical terms on first use

2. Consistency
   - Use consistent terminology
   - Maintain consistent formatting
   - Follow established patterns

3. Structure
   - Use hierarchical headings
   - Keep sections focused
   - Include examples

## Formatting Standards

### Markdown Usage

```markdown
# Top-level heading
## Second-level heading
### Third-level heading

- Bullet points for lists
- Use dashes

1. Numbered lists for sequences
2. Use periods

`inline code`

```python
# Code blocks with language
def example():
    pass
```
```

### Code Examples

- Include language identifier
- Use meaningful variable names
- Add comments for clarity
- Show complete, runnable examples

```python
# Good example
from llm_consortium import ConsortiumOrchestrator

async def process_query(prompt: str) -> dict:
    """Process a query using the consortium."""
    orchestrator = ConsortiumOrchestrator()
    return await orchestrator.orchestrate(prompt)
```

### Links and References

- Use relative links for internal documentation
- Use descriptive link text
- Include section anchors when needed

```markdown
[See Configuration Guide](DEPLOYMENT.md#configuration)
```

## Content Guidelines

### Document Structure

1. Title
2. Brief description
3. Prerequisites (if any)
4. Main content
5. Related resources
6. Examples
7. Troubleshooting (if applicable)

### Writing Style

- Use active voice
- Keep paragraphs short
- Use appropriate technical level
- Include practical examples
- Add context when needed

### Code Documentation

1. Function/Class Documentation
```python
def function_name(param: type) -> return_type:
    """Brief description.

    Detailed description.

    Args:
        param: Description

    Returns:
        Description

    Raises:
        ErrorType: Description
    """
```

2. Module Documentation
```python
"""
Module Name

Brief description.

Detailed description.

Examples:
    Basic usage example

Attributes:
    module_level_variable: Description
"""
```

## File Organization

### Directory Structure
```
docs/
├── INDEX.md
├── QUICK_START.md
├── technical/
│   ├── ARCHITECTURE.md
│   └── API.md
├── guides/
│   ├── DEPLOYMENT.md
│   └── DEVELOPMENT.md
└── reference/
    ├── TROUBLESHOOTING.md
    └── SECURITY.md
```

### File Naming
- Use UPPERCASE for documentation files
- Use lowercase with underscores for code files
- Include purpose in filename

## Version Documentation

### Version Tags
```markdown
> Available since version 0.1.0
```

### Deprecation Notices
```markdown
!!! warning "Deprecated"
    This feature is deprecated since version 0.2.0
```

## Best Practices

1. Keep Documentation Current
   - Update with code changes
   - Review regularly
   - Mark outdated sections

2. Use Templates
   - Follow standard templates
   - Maintain consistency
   - Include all required sections

3. Include Examples
   - Show common use cases
   - Provide complete code
   - Explain expected output

4. Consider Accessibility
   - Use alt text for images
   - Provide text alternatives
   - Maintain good contrast

5. Document Errors
   - Include error messages
   - Explain causes
   - Provide solutions
