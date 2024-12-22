import asyncio
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from typing import List

from consortium import ConsortiumPlugin, ConsortiumResponse

class TestConsortiumPlugin(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.plugin = ConsortiumPlugin()
        # Override history file location for testing
        self.plugin.history_file = os.path.join(self.test_dir, "test_history.csv")

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    async def async_test_get_model_response(self):
        """Test getting response from a single model"""
        model_name = "test-model"
        prompt = "Test prompt"
        
        # Mock the LLM model response
        mock_response = Mock()
        mock_response.__str__.return_value = "<confidence>0.95</confidence>\nTest response"
        
        with patch('llm.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.complete_async.return_value = mock_response
            mock_get_model.return_value = mock_model
            
            response = await self.plugin.get_model_response(model_name, prompt)
            
            self.assertEqual(response.model_name, model_name)
            self.assertEqual(response.confidence, 0.95)
            self.assertIn("Test response", response.response)

    async def async_test_get_arbiter_decision(self):
        """Test arbiter decision making"""
        responses = [
            ConsortiumResponse("model1", "Response 1", 0.8, {}),
            ConsortiumResponse("model2", "Response 2", 0.7, {})
        ]
        prompt = "Test prompt"
        
        mock_response = Mock()
        mock_response.__str__.return_value = """<confidence>0.95</confidence>
<response>
Synthesized response
</response>"""
        
        with patch('llm.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.complete_async.return_value = mock_response
            mock_get_model.return_value = mock_model
            
            response = await self.plugin.get_arbiter_decision("arbiter-model", responses, prompt)
            
            self.assertEqual(response.confidence, 0.95)
            self.assertIn("Synthesized response", response.response)

    async def async_test_run_consortium(self):
        """Test full consortium run"""
        prompt = "Test prompt"
        models = ["model1", "model2"]
        
        # Mock responses for individual models
        mock_responses = [
            "<confidence>0.8</confidence>\nResponse 1",
            "<confidence>0.7</confidence>\nResponse 2"
        ]
        
        # Mock arbiter response
        mock_arbiter_response = """<confidence>0.95</confidence>
<response>
Final arbitrated response
</response>"""
        
        with patch('llm.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.complete_async.side_effect = [
                Mock(__str__=lambda self: resp) for resp in mock_responses + [mock_arbiter_response]
            ]
            mock_get_model.return_value = mock_model
            
            response = await self.plugin.run_consortium(
                prompt=prompt,
                models=models,
                arbiter_model="arbiter-model",
                confidence_threshold=0.9,
                max_iterations=2
            )
            
            self.assertEqual(response.confidence, 0.95)
            self.assertIn("Final arbitrated response", response.response)

    def test_history_logging(self):
        """Test that interactions are properly logged"""
        self.plugin.log_interaction(
            prompt="Test prompt",
            models=["model1", "model2"],
            arbiter="arbiter-model",
            iterations=1,
            response="Test response",
            confidence=0.95
        )
        
        # Verify log file exists and contains the entry
        self.assertTrue(os.path.exists(self.plugin.history_file))
        with open(self.plugin.history_file, 'r') as f:
            content = f.read()
            self.assertIn("Test prompt", content)
            self.assertIn("model1,model2", content)
            self.assertIn("Test response", content)
            self.assertIn("0.95", content)

    def test_cli(self):
        """Test CLI interface"""
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = ConsortiumResponse(
                model_name="test-model",
                response="CLI test response",
                confidence=0.95,
                metadata={}
            )
            
            with patch('sys.stdout') as mock_stdout:
                self.plugin.cli([
                    "Test prompt",
                    "-m", "model1",
                    "-m", "model2",
                    "--arbiter-model", "arbiter-model",
                    "--confidence-threshold", "0.9",
                    "--max-iterations", "2"
                ])
                
                mock_run.assert_called_once()

def run_async_test(coro):
    return asyncio.run(coro)

if __name__ == '__main__':
    # Run async tests
    test_instance = TestConsortiumPlugin()
    test_instance.setUp()
    
    try:
        run_async_test(test_instance.async_test_get_model_response())
        run_async_test(test_instance.async_test_get_arbiter_decision())
        run_async_test(test_instance.async_test_run_consortium())
        
        # Run synchronous tests
        test_instance.test_history_logging()
        test_instance.test_cli()
        
        print("All tests passed!")
    finally:
        test_instance.tearDown()
