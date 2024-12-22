# Context Manager Implementation Proposal

## Overview
This document outlines the specific implementation details for the Context Management System, which has been identified as the highest priority improvement for the LLM Karpathy Consortium.

## Core Implementation

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class ModelResponse:
    model_name: str
    response: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class IterationData:
    iteration_number: int
    timestamp: str
    synthesis: Dict
    model_responses: List[ModelResponse]
    refinement_areas: List[str]
    metrics: Dict = field(default_factory=dict)

class ContextManager:
    def __init__(
        self, 
        original_prompt: str, 
        document_context: Optional[str] = None,
        max_history_length: int = 3,
        context_compression_threshold: int = 1000
    ):
        self.original_prompt = original_prompt
        self.document_context = document_context
        self.iteration_history: List[IterationData] = []
        self.max_history_length = max_history_length
        self.context_compression_threshold = context_compression_threshold
        
    def add_iteration(self, iteration_data: Dict) -> None:
        """Add a new iteration to the history with proper structure and validation."""
        model_responses = [
            ModelResponse(**resp) for resp in iteration_data['model_responses']
        ]
        
        new_iteration = IterationData(
            iteration_number=len(self.iteration_history) + 1,
            timestamp=datetime.now().isoformat(),
            synthesis=iteration_data['synthesis'],
            model_responses=model_responses,
            refinement_areas=iteration_data['refinement_areas'],
            metrics=self._calculate_iteration_metrics(iteration_data)
        )
        
        self.iteration_history.append(new_iteration)
        self._maybe_compress_history()

    def construct_prompt(self) -> str:
        """Construct a comprehensive prompt with proper context management."""
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

    def _calculate_iteration_metrics(self, iteration_data: Dict) -> Dict:
        """Calculate metrics for the iteration including confidence trends and convergence indicators."""
        return {
            "average_confidence": sum(r.confidence for r in iteration_data['model_responses']) / len(iteration_data['model_responses']),
            "consensus_level": self._calculate_consensus_level(iteration_data['model_responses']),
            "refinement_progress": self._calculate_refinement_progress(iteration_data['refinement_areas'])
        }

    def _get_formatted_history(self) -> List[Dict]:
        """Format iteration history for prompt inclusion, with smart summarization."""
        formatted_history = []
        for iteration in self.iteration_history[-self.max_history_length:]:
            formatted_history.append({
                "iteration": iteration.iteration_number,
                "synthesis": iteration.synthesis,
                "key_points": self._extract_key_points(iteration),
                "refinement_areas": iteration.refinement_areas,
                "metrics": iteration.metrics
            })
        return formatted_history

    def _format_prompt(self, context: Dict) -> str:
        """Format the final prompt with clear structure and emphasis on key areas."""
        return f"""Original Request:
{context['original_request']['prompt']}

Document Context:
{context['original_request']['document_context'] or 'None provided'}

Previous Iterations Summary:
{json.dumps(context['iteration_history'], indent=2)}

Current Status:
- Iteration: {context['current_state']['iteration_number']}
- Focus Areas: {', '.join(context['current_state']['current_focus'])}
- Progress: {json.dumps(context['current_state']['progress_metrics'], indent=2)}

Please provide an updated response that:
1. Addresses the current focus areas
2. Builds upon previous insights
3. Maintains alignment with the original request
"""

    def _maybe_compress_history(self) -> None:
        """Compress history if it exceeds thresholds while maintaining context integrity."""
        if len(self.iteration_history) > self.max_history_length:
            # Implement smart compression logic here
            # For now, just keep the most recent iterations
            self.iteration_history = self.iteration_history[-self.max_history_length:]
```

## Integration Plan

1. **Initial Integration**
```python
# In ConsortiumOrchestrator

def __init__(self, models, arbiter_model, **kwargs):
    self.context_manager = ContextManager(
        max_history_length=kwargs.get('max_history_length', 3),
        context_compression_threshold=kwargs.get('context_compression_threshold', 1000)
    )
    # ... rest of current init ...

def orchestrate(self, prompt: str, document_context: Optional[str] = None):
    self.context_manager = ContextManager(prompt, document_context)
    
    while True:
        current_prompt = self.context_manager.construct_prompt()
        responses = self._get_model_responses(current_prompt)
        synthesis = self._get_arbiter_synthesis(responses)
        
        self.context_manager.add_iteration({
            'model_responses': responses,
            'synthesis': synthesis,
            'refinement_areas': synthesis.get('refinement_areas', [])
        })
        
        if not synthesis.get('needs_iteration', False):
            break
```

## Next Steps

1. **Implementation Phases**
   - Phase 1: Basic Context Management
   - Phase 2: Add Metrics and Analytics
   - Phase 3: Implement Smart Compression
   - Phase 4: Add Configuration Options

2. **Testing Strategy**
   - Unit tests for each ContextManager method
   - Integration tests with ConsortiumOrchestrator
   - Performance tests for large histories
   - Validation of prompt construction

3. **Documentation Needs**
   - API documentation
   - Configuration guide
   - Best practices
   - Example usage

Would you like me to proceed with implementing any specific part of this proposal?
