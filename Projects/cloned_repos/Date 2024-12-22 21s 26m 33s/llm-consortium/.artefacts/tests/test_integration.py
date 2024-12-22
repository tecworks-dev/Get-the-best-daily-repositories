"""
Integration tests for LLM Karpathy Consortium
"""
import pytest
import asyncio
import os
from pathlib import Path
from llm_consortium import ConsortiumOrchestrator, DatabaseConnection, setup_logging

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_orchestration_flow():
    """Test the complete orchestration flow with actual model calls."""
    setup_logging()
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229"],  # Using single model for faster testing
        confidence_threshold=0.8,
        max_iterations=2
    )
    
    result = await orchestrator.orchestrate("What is 2+2?")
    
    assert result["metadata"]["iteration_count"] >= 1
    assert "synthesis" in result
    assert result["synthesis"]["confidence"] > 0

@pytest.mark.integration
def test_database_integration():
    """Test database operations in a real environment."""
    # Setup test database
    test_db_path = Path("./test_data/test_logs.db")
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Override database path for testing
    original_path = os.environ.get("LLM_USER_PATH")
    os.environ["LLM_USER_PATH"] = str(test_db_path.parent)
    
    try:
        db = DatabaseConnection.get_connection()
        
        # Test table creation
        db["responses"].insert({
            "model": "test-model",
            "prompt": "test prompt",
            "response": "test response",
            "timestamp": "2024-01-01T00:00:00"
        })
        
        # Verify data
        assert len(db["responses"].rows) > 0
        
    finally:
        # Cleanup
        if original_path:
            os.environ["LLM_USER_PATH"] = original_path
        else:
            del os.environ["LLM_USER_PATH"]
        
        if test_db_path.exists():
            test_db_path.unlink()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multimodel_integration():
    """Test orchestration with multiple models."""
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229", "gpt-4"],
        confidence_threshold=0.9,
        max_iterations=2
    )
    
    result = await orchestrator.orchestrate(
        "Explain the difference between quantum and classical computing in one sentence."
    )
    
    assert len(result["model_responses"]) == 2
    assert all("response" in r for r in result["model_responses"])
    assert "synthesis" in result

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_iterative_refinement():
    """Test the iterative refinement process with a complex prompt."""
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229"],
        confidence_threshold=0.95,  # High threshold to force iteration
        max_iterations=3
    )
    
    result = await orchestrator.orchestrate(
        """Analyze the implications of using quantum computing for:
        1. Cryptography
        2. Drug discovery
        3. Climate modeling
        Provide specific examples and potential timeline."""
    )
    
    assert result["metadata"]["iteration_count"] > 1
    assert result["synthesis"]["confidence"] > 0
    assert len(result["synthesis"]["analysis"]) > 0

@pytest.mark.integration
def test_logging_integration():
    """Test logging system integration."""
    test_log_path = Path("./test_data/test.log")
    test_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging for test
    setup_logging()
    
    try:
        # Trigger some logging
        orchestrator = ConsortiumOrchestrator(
            models=["claude-3-sonnet-20240229"]
        )
        
        assert test_log_path.parent.exists()
        
    finally:
        # Cleanup
        if test_log_path.exists():
            test_log_path.unlink()
