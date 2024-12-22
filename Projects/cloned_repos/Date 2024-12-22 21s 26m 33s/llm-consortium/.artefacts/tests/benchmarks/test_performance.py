"""
Performance benchmarks for LLM Karpathy Consortium
"""
import pytest
import asyncio
import time
from llm_consortium import ConsortiumOrchestrator

@pytest.mark.benchmark
async def test_single_model_performance(benchmark):
    """Benchmark single model performance."""
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229"]
    )
    
    async def run_orchestration():
        return await orchestrator.orchestrate("What is 2+2?")
    
    result = await benchmark(run_orchestration)
    assert result is not None

@pytest.mark.benchmark
async def test_multi_model_performance(benchmark):
    """Benchmark multi-model performance."""
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229", "gpt-4"]
    )
    
    async def run_orchestration():
        return await orchestrator.orchestrate(
            "Explain the concept of quantum entanglement."
        )
    
    result = await benchmark(run_orchestration)
    assert result is not None

@pytest.mark.benchmark
async def test_database_performance(benchmark):
    """Benchmark database operations."""
    from llm_consortium import DatabaseConnection, log_response
    from unittest.mock import Mock
    
    mock_response = Mock()
    mock_response.text.return_value = "Test response"
    
    def db_operation():
        log_response(mock_response, "test-model")
    
    benchmark(db_operation)

@pytest.mark.benchmark
async def test_response_parsing_performance(benchmark):
    """Benchmark response parsing performance."""
    orchestrator = ConsortiumOrchestrator(
        models=["claude-3-sonnet-20240229"]
    )
    
    test_response = """
    <synthesis_output>
        <synthesis>Test synthesis</synthesis>
        <confidence>0.9</confidence>
        <analysis>Test analysis</analysis>
        <dissent>Test dissent</dissent>
        <needs_iteration>false</needs_iteration>
        <refinement_areas>Test refinement</refinement_areas>
    </synthesis_output>
    """
    
    def parse_response():
        return orchestrator._parse_arbiter_response(test_response)
    
    result = benchmark(parse_response)
    assert result is not None
