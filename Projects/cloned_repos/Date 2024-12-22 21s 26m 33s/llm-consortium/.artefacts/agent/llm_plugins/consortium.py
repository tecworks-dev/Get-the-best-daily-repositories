"""LLM Plugin for managing a consortium of language models with arbitration."""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import csv
from datetime import datetime
import os
import functools
from abc import ABC, abstractmethod

import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConsortiumResponse:
    """Represents a response from a model in the consortium."""
    model_name: str
    response: str
    confidence: float
    metadata: Dict[Any, Any]
    latency: float = 0.0  # Response time in seconds
    tokens_used: int = 0   # Number of tokens used

class ResponseParser:
    """Handles parsing and validation of model responses."""
    
    @staticmethod
    def extract_confidence(response_text: str) -> float:
        """Extract confidence score from response text."""
        try:
            if '<confidence>' in response_text and '</confidence>' in response_text:
                confidence_str = response_text.split('<confidence>')[1].split('</confidence>')[0]
                confidence = float(confidence_str)
                return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        except Exception as e:
            logger.warning(f"Failed to extract confidence: {e}")
        return 0.0

    @staticmethod
    def extract_response_content(response_text: str) -> str:
        """Extract main response content from response text."""
        try:
            if '<response>' in response_text and '</response>' in response_text:
                return response_text.split('<response>')[1].split('</response>')[0].strip()
        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
        return response_text

def cache_response(ttl_seconds: int = 3600):
    """Decorator to cache model responses."""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{args}:{kwargs}"
            
            # Check cache
            if key in cache:
                timestamp, response = cache[key]
                if (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                    return response
                
            # Get fresh response
            response = await func(*args, **kwargs)
            cache[key] = (datetime.now(), response)
            return response
        return wrapper
    return decorator

class VotingStrategy(ABC):
    """Abstract base class for voting strategies."""
    
    @abstractmethod
    def compute_weights(self, responses: List[ConsortiumResponse]) -> List[float]:
        """Compute weights for each response."""
        pass

class ConfidenceWeightedVoting(VotingStrategy):
    """Weights responses based on confidence scores."""
    
    def compute_weights(self, responses: List[ConsortiumResponse]) -> List[float]:
        confidences = [r.confidence for r in responses]
        total = sum(confidences) or 1.0  # Avoid division by zero
        return [c / total for c in confidences]

class ConsortiumMetrics:
    """Tracks and records consortium performance metrics."""
    
    def __init__(self, metrics_file: str = "consortium_metrics.csv"):
        self.metrics_file = metrics_file
        self._ensure_metrics_file()
        
    def _ensure_metrics_file(self):
        """Ensure metrics file exists with headers."""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'model', 'latency', 'confidence', 
                               'tokens_used', 'success'])

    def log_metric(self, model: str, latency: float, confidence: float,
                  tokens_used: int, success: bool):
        """Log a single metric entry."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                model,
                latency,
                confidence,
                tokens_used,
                success
            ])

class ConsortiumPlugin(llm.Plugin):
    """Plugin for managing a consortium of language models with arbitration."""
    
    name = 'consortium'

    def __init__(self):
        self.history_file = "consortium_history.csv"
        self.parser = ResponseParser()
        self.metrics = ConsortiumMetrics()
        self.voting_strategy = ConfidenceWeightedVoting()
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Ensure the history file exists with headers."""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'prompt', 'models', 'arbiter', 
                               'iterations', 'final_response', 'confidence',
                               'total_latency', 'total_tokens'])

    def log_interaction(self, prompt: str, models: List[str], arbiter: str,
                       iterations: int, response: str, confidence: float,
                       latency: float, tokens: int):
        """Log consortium interaction to CSV file."""
        with open(self.history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                prompt,
                ','.join(models),
                arbiter,
                iterations,
                response,
                confidence,
                latency,
                tokens
            ])

    @cache_response(ttl_seconds=3600)
    async def get_model_response(self, model_name: str, prompt: str) -> ConsortiumResponse:
        """Get response from a specific model."""
        start_time = datetime.now()
        try:
            model = llm.get_model(model_name)
            response = await model.complete_async(prompt)
            
            latency = (datetime.now() - start_time).total_seconds()
            response_text = str(response)
            
            # Parse response
            confidence = self.parser.extract_confidence(response_text)
            clean_response = self.parser.extract_response_content(response_text)
            
            # Get token usage if available
            tokens_used = getattr(response, 'tokens_used', 0)
            
            result = ConsortiumResponse(
                model_name=model_name,
                response=clean_response,
                confidence=confidence,
                metadata={'raw_response': response},
                latency=latency,
                tokens_used=tokens_used
            )
            
            # Log metrics
            self.metrics.log_metric(
                model=model_name,
                latency=latency,
                confidence=confidence,
                tokens_used=tokens_used,
                success=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting response from {model_name}: {e}")
            latency = (datetime.now() - start_time).total_seconds()
            
            # Log failed attempt
            self.metrics.log_metric(
                model=model_name,
                latency=latency,
                confidence=0.0,
                tokens_used=0,
                success=False
            )
            
            return ConsortiumResponse(
                model_name=model_name,
                response=f"Error: {str(e)}",
                confidence=0.0,
                metadata={'error': str(e)},
                latency=latency,
                tokens_used=0
            )

    async def get_arbiter_decision(self, arbiter_model: str, responses: List[ConsortiumResponse],
                                 original_prompt: str) -> ConsortiumResponse:
        """Get arbitration decision from the arbiter model."""
        start_time = datetime.now()
        
        # Apply voting strategy
        weights = self.voting_strategy.compute_weights(responses)
        
        arbiter = llm.get_model(arbiter_model)
        
        arbitration_prompt = f"""
As an arbiter, analyze these model responses and choose the best one or synthesize a better response.
Original prompt: {original_prompt}

Responses with weights:
{json.dumps([{
    'model': r.model_name,
    'response': r.response,
    'confidence': r.confidence,
    'weight': w
} for r, w in zip(responses, weights)], indent=2)}

Provide your decision in this format:
<confidence>0.95</confidence>
<response>
Your chosen or synthesized response here
</response>
"""
        
        response = await arbiter.complete_async(arbitration_prompt)
        latency = (datetime.now() - start_time).total_seconds()
        
        response_text = str(response)
        confidence = self.parser.extract_confidence(response_text)
        clean_response = self.parser.extract_response_content(response_text)
        
        return ConsortiumResponse(
            model_name=arbiter_model,
            response=clean_response,
            confidence=confidence,
            metadata={'raw_response': response},
            latency=latency,
            tokens_used=getattr(response, 'tokens_used', 0)
        )

    def cli(self, args: List[str]) -> None:
        """CLI interface for the consortium plugin."""
        import argparse
        
        parser = argparse.ArgumentParser(description="Run a prompt through a model consortium")
        parser.add_argument("prompt", help="The prompt to process")
        parser.add_argument("-m", "--model", action="append", required=True,
                          help="Model to use (can be specified multiple times)")
        parser.add_argument("--arbiter-model", default="claude-3-sonnet-20240307",
                          help="Model to use for arbitration")
        parser.add_argument("--confidence-threshold", type=float, default=0.9,
                          help="Confidence threshold for accepting responses")
        parser.add_argument("--max-iterations", type=int, default=2,
                          help="Maximum number of iteration attempts")
        parser.add_argument("--no-cache", action="store_true",
                          help="Disable response caching")
        
        parsed_args = parser.parse_args(args)
        
        # Run the consortium
        result = asyncio.run(self.run_consortium(
            prompt=parsed_args.prompt,
            models=parsed_args.model,
            arbiter_model=parsed_args.arbiter_model,
            confidence_threshold=parsed_args.confidence_threshold,
            max_iterations=parsed_args.max_iterations,
            use_cache=not parsed_args.no_cache
        ))
        
        print(json.dumps({
            'response': result.response,
            'confidence': result.confidence,
            'model': result.model_name,
            'latency': result.latency,
            'tokens_used': result.tokens_used
        }, indent=2))

    async def run_consortium(self, prompt: str, models: List[str],
                           arbiter_model: str, confidence_threshold: float,
                           max_iterations: int, use_cache: bool = True) -> ConsortiumResponse:
        """Run the consortium process."""
        start_time = datetime.now()
        iteration = 0
        final_response = None
        total_tokens = 0
        
        while iteration < max_iterations:
            # Get responses from all models
            get_response_func = self.get_model_response if use_cache else \
                              functools.partial(self.get_model_response.__wrapped__, self)
            
            responses = await asyncio.gather(
                *[get_response_func(model, prompt) for model in models]
            )
            
            # Update token count
            total_tokens += sum(r.tokens_used for r in responses)
            
            # Check if any response meets the confidence threshold
            high_confidence_responses = [
                r for r in responses if r.confidence >= confidence_threshold
            ]
            
            if high_confidence_responses:
                final_response = max(high_confidence_responses,
                                   key=lambda x: x.confidence)
                break
            
            # Get arbitration if no high confidence responses
            final_response = await self.get_arbiter_decision(
                arbiter_model, responses, prompt
            )
            
            total_tokens += final_response.tokens_used
            
            if final_response.confidence >= confidence_threshold:
                break
                
            iteration += 1
            
            # Update prompt for next iteration if needed
            prompt = f"""Previous responses were not confident enough. 
Original prompt: {prompt}
Please provide a more detailed and confident response."""
        
        total_latency = (datetime.now() - start_time).total_seconds()
        
        # Log the interaction
        self.log_interaction(
            prompt=prompt,
            models=models,
            arbiter=arbiter_model,
            iterations=iteration + 1,
            response=final_response.response,
            confidence=final_response.confidence,
            latency=total_latency,
            tokens=total_tokens
        )
        
        return final_response

# Register the plugin
registry = [ConsortiumPlugin]
