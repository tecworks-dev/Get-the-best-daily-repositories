# Testing LLM Karpathy Consortium

This directory contains the test suite for the LLM Karpathy Consortium project.

## Test Structure

```
tests/
├── __init__.py
├── test_orchestrator.py
├── test_database.py
├── test_responses.py
└── fixtures/
    └── sample_responses.json
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=llm_consortium

# Run specific test file
pytest tests/test_orchestrator.py

# Run tests with detailed output
pytest -v
```

## Writing Tests

Follow these guidelines when adding new tests:

1. Use meaningful test names that describe the behavior being tested
2. Follow the Arrange-Act-Assert pattern
3. Use fixtures for common setup
4. Mock external services and API calls
5. Include both positive and negative test cases
