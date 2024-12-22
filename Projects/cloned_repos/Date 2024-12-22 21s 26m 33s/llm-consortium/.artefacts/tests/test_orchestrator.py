"""
Tests for the ConsortiumOrchestrator class.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from llm_consortium import ConsortiumOrchestrator

@pytest.fixture
def orchestrator():
    return ConsortiumOrchestrator(
        models=["test-model-1", "test-model-2"],
        confidence_threshold=0.8,
        max_iterations=2
    )

@pytest.fixture
def mock_response():
    return {
        "text": """
        <thought_process>Test thinking</thought_process>
        <answer>Test answer</answer>
        <confidence>0.85</confidence>
        """
    }

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization with default values."""
    assert len(orchestrator.models) == 2
    assert orchestrator.confidence_threshold == 0.8
    assert orchestrator.max_iterations == 2
    assert orchestrator.arbiter_model == "claude-3-opus-20240229"

@pytest.mark.asyncio
async def test_orchestrate_success(orchestrator, mock_response):
    """Test successful orchestration with mock responses."""
    with patch('llm.get_model') as mock_get_model:
        mock_model = Mock()
        mock_model.prompt.return_value.text.return_value = mock_response["text"]
        mock_get_model.return_value = mock_model

        result = await orchestrator.orchestrate("Test prompt")
        
        assert result["metadata"]["iteration_count"] == 1
        assert "synthesis" in result
        assert "model_responses" in result

@pytest.mark.asyncio
async def test_confidence_threshold(orchestrator):
    """Test behavior when confidence threshold is not met."""
    with patch('llm.get_model') as mock_get_model:
        mock_model = Mock()
        mock_model.prompt.return_value.text.return_value = """
        <thought_process>Test thinking</thought_process>
        <answer>Test answer</answer>
        <confidence>0.5</confidence>
        """
        mock_get_model.return_value = mock_model

        result = await orchestrator.orchestrate("Test prompt")
        
        # Should attempt multiple iterations due to low confidence
        assert result["metadata"]["iteration_count"] > 1

@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    """Test error handling during orchestration."""
    with patch('llm.get_model') as mock_get_model:
        mock_model = Mock()
        mock_model.prompt.side_effect = Exception("Test error")
        mock_get_model.return_value = mock_model

        with pytest.raises(Exception) as exc_info:
            await orchestrator.orchestrate("Test prompt")
        
        assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_response_parsing(orchestrator):
    """Test parsing of structured responses."""
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
    
    with patch('llm.get_model') as mock_get_model:
        mock_model = Mock()
        mock_model.prompt.return_value.text.return_value = test_response
        mock_get_model.return_value = mock_model

        result = await orchestrator.orchestrate("Test prompt")
        
        assert result["synthesis"]["synthesis"] == "Test synthesis"
        assert result["synthesis"]["confidence"] == 0.9
        assert result["synthesis"]["analysis"] == "Test analysis"
