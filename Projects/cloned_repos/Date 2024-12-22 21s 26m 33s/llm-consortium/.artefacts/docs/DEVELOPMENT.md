# Development Guide

This document outlines the development workflow and best practices for contributing to the LLM Karpathy Consortium project.

## Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-karpathy-consortium.git
cd llm-karpathy-consortium
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Quality Tools

### Pre-commit Hooks
The project uses pre-commit hooks to ensure code quality:
- Black for code formatting
- Flake8 for style guide enforcement
- MyPy for type checking
- Various file checks

To run manually:
```bash
pre-commit run --all-files
```

### Testing
Tests are written using pytest:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test category
pytest -m integration
pytest -m benchmark
```

### Type Checking
MyPy is configured for strict type checking:
```bash
mypy llm_consortium
```

## CI/CD Pipeline

The GitHub Actions workflow (.github/workflows/tests.yml) includes:

1. Test runs on multiple Python versions
2. Code coverage reporting
3. Linting checks
4. Performance benchmarks

### Coverage Reports
Coverage reports are generated in multiple formats:
- Terminal output
- HTML report
- XML for Codecov

View HTML coverage report:
```bash
pytest --cov=llm_consortium --cov-report=html
# Open coverage_html/index.html in browser
```

## Benchmarking

Run performance benchmarks:
```bash
pytest tests/benchmarks/test_performance.py -v
```

## Documentation

- Keep docstrings up to date (Google style)
- Update README.md for user-facing changes
- Update CHANGELOG.md following semver
- Add migration guides for breaking changes

## Release Process

1. Update version in __init__.py
2. Update CHANGELOG.md
3. Create and push tag:
```bash
git tag -a v0.x.x -m "Release v0.x.x"
git push origin v0.x.x
```

## Best Practices

1. Write tests for new features
2. Maintain type hints
3. Follow PEP 8 style guide
4. Keep functions focused and documented
5. Use meaningful variable names
6. Handle errors gracefully
7. Log important operations

## Troubleshooting

Common issues and solutions:

1. Pre-commit hooks failing:
   ```bash
   pre-commit clean
   pre-commit run --all-files
   ```

2. Test database issues:
   ```bash
   rm -rf test_data/
   pytest --create-db
   ```

3. Coverage report issues:
   ```bash
   coverage erase
   pytest --cov
   ```

## Getting Help

- Create an issue for bugs
- Use discussions for questions
- Join developer chat for real-time help

