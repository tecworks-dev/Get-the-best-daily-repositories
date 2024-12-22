# Contributing to LLM Karpathy Consortium

Thank you for your interest in contributing to the LLM Karpathy Consortium project! This document provides guidelines and information for contributors.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/irthomasthomas/llm-consortium.git
cd llm-consortium
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Document classes and functions using docstrings
- Keep functions focused and single-purpose
- Use meaningful variable and function names

## Testing

- Write tests for new features using pytest
- Ensure all tests pass before submitting PR
- Maintain or improve code coverage
- Run tests using:
```bash
pytest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/description`)
3. Make your changes
4. Write or update tests
5. Update documentation
6. Commit with clear, descriptive messages
7. Push to your fork
8. Submit a Pull Request

## Documentation

- Update README.md if adding new features
- Add docstrings for new functions and classes
- Update CHANGELOG.md with your changes
- Include examples for new functionality

## Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce
- Include expected vs actual behavior
- Include Python version and environment details
- Include relevant logs or stack traces

## Code Review Process

1. At least one maintainer must review and approve
2. All tests must pass
3. Documentation must be updated
4. Code style must follow project guidelines

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
