"""
Context Management System for LLM Karpathy Consortium
Handles preservation and structuring of context across iterations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Structured container for individual model responses"""
    model_name: str
    response: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "model_name": self.model_name,
            "response": self.response,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }

@dataclass
class IterationData:
    """Container for data from a single iteration"""
    iteration_number: int
    timestamp: str
    synthesis: Dict
    model_responses: List[ModelResponse]
    refinement_areas: List[str]
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp,
            "synthesis": self.synthesis,
            "model_responses": [r.to_dict() for r in self.model_responses],
            "refinement_areas": self.refinement_areas,
            "metrics": self.metrics
        }

class ContextManager:
    """Manages context preservation and structuring across consortium iterations"""
    
    def __init__(
        self, 
        original_prompt: str, 
        document_context: Optional[str] = None,
        max_history_length: int = 3,
        context_compression_threshold: int = 1000
    ):
        """Initialize the context manager"""
        self.original_prompt = original_prompt
        self.document_context = document_context
        self.iteration_history: List[IterationData] = []
        self.max_history_length = max_history_length
        self.context_compression_threshold = context_compression_threshold
        logger.info("Initialized ContextManager with max_history_length=%d", max_history_length)

    def add_iteration(self, iteration_data: Dict) -> None:
        """Add a new iteration to the history"""
        try:
            # Convert raw model response dictionaries to ModelResponse objects
            model_responses = [
                ModelResponse(
                    model_name=resp['model_name'],
                    response=resp['response'],
                    confidence=resp['confidence'],
                    reasoning=resp.get('reasoning'),
                    metadata=resp.get('metadata', {})
                ) for resp in iteration_data['model_responses']
            ]
            
            # Calculate metrics before creating IterationData
            metrics = self._calculate_iteration_metrics(model_responses)
            
            new_iteration = IterationData(
                iteration_number=len(self.iteration_history) + 1,
                timestamp=datetime.now().isoformat(),
                synthesis=iteration_data['synthesis'],
                model_responses=model_responses,
                refinement_areas=iteration_data['refinement_areas'],
                metrics=metrics
            )
            
            self.iteration_history.append(new_iteration)
            self._maybe_compress_history()
            logger.info("Added iteration %d to history", new_iteration.iteration_number)
            
        except Exception as e:
            logger.error("Failed to add iteration: %s", str(e))
            raise

    def construct_prompt(self) -> str:
        """Construct a comprehensive prompt with proper context management"""
        try:
            context = {
                "original_request": {
                    "prompt": self.original_prompt,
                    "document_context": self.document_context
                },
                "iteration_history": self._get_formatted_history(),
                "current_state": {
                    "iteration_number": len(self.iteration_history),
                    "current_focus": self._get_current_refinement_areas(),
                    "progress_metrics": self._get_progress_metrics()
                }
            }
            
            return self._format_prompt(context)
            
        except Exception as e:
            logger.error("Failed to construct prompt: %s", str(e))
            raise

    def _calculate_iteration_metrics(self, model_responses: List[ModelResponse]) -> Dict:
        """Calculate metrics for the current iteration"""
        try:
            confidences = [resp.confidence for resp in model_responses]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "average_confidence": avg_confidence,
                "num_models": len(model_responses),
                "confidence_range": {
                    "min": min(confidences) if confidences else 0,
                    "max": max(confidences) if confidences else 0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning("Failed to calculate metrics: %s", str(e))
            return {}

    def _get_formatted_history(self) -> List[Dict]:
        """Format iteration history for prompt inclusion"""
        return [
            iteration.to_dict() 
            for iteration in self.iteration_history[-self.max_history_length:]
        ]

    def _get_current_refinement_areas(self) -> List[str]:
        """Get current refinement areas from last iteration"""
        if not self.iteration_history:
            return []
        return self.iteration_history[-1].refinement_areas

    def _get_progress_metrics(self) -> Dict:
        """Calculate overall progress metrics"""
        if not self.iteration_history:
            return {}
            
        confidence_trend = [
            iteration.metrics.get('average_confidence', 0)
            for iteration in self.iteration_history
        ]
        
        return {
            "total_iterations": len(self.iteration_history),
            "confidence_trend": confidence_trend,
            "current_confidence": confidence_trend[-1] if confidence_trend else 0
        }

    def _format_prompt(self, context: Dict) -> str:
        """Format the final prompt with clear structure"""
        return f"""Original Request:
{context['original_request']['prompt']}

Document Context:
{context['original_request']['document_context'] or 'None provided'}

Previous Iterations Summary:
{json.dumps(self._get_formatted_history(), indent=2)}

Current Status:
- Iteration: {context['current_state']['iteration_number']}
- Focus Areas: {', '.join(context['current_state']['current_focus'])}
- Progress Metrics: {json.dumps(context['current_state']['progress_metrics'], indent=2)}

Please provide an updated response that:
1. Addresses the current focus areas
2. Builds upon previous insights
3. Maintains alignment with the original request
"""

    def _maybe_compress_history(self) -> None:
        """Compress history if it exceeds thresholds"""
        if len(self.iteration_history) > self.max_history_length:
            logger.info("Compressing history to maintain max_history_length")
            self.iteration_history = self.iteration_history[-self.max_history_length:]
