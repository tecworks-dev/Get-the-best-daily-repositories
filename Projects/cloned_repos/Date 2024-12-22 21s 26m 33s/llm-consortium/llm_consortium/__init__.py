import click
import json
import llm
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import sys
import re
import os
import pathlib
import sqlite_utils

# Todo: Add a parser-model llm - if confidence score, or anything else is not found in the response by normal parsing - try to parse it with the parser model. If that fails, try to prompt the original model again n times.
# Todo: <iteration_history> is empty.
DEFAULT_SYSTEM_PROMPT = """Please follow these steps when formulating your response:

1. Begin by carefully considering the specific instructions provided.

2. Write your thought process inside <thought_process> tags. In this section:
   - List key aspects that are relevant to the query
   - Identify potential challenges or limitations
   - Consider how the response instructions affect your approach
   - Explore different angles, consider potential challenges, and explain your logic step-by-step

3. After your thought process, provide your confidence level on a scale from 0 to 1, where 0 represents no confidence and 1 represents absolute certainty. Use <confidence> tags for this.

4. Finally, present your answer within <answer> tags.

Your response should follow this structure:

<thought_process>
[Your detailed thought process, exploring various aspects of the problem]
</thought_process>

<answer>
[Your final, well-considered answer to the query]
</answer>

<confidence>
[Your confidence level from 0 to 1]
</confidence>

Remember to be thorough in your reasoning, clear in your explanations, and precise in your confidence assessment. Your contribution is valuable to the consortium's collaborative problem-solving efforts."""

def user_dir() -> pathlib.Path:
    """Get or create user directory for storing application data."""
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path

def logs_db_path() -> pathlib.Path:
    """Get path to logs database."""
    return user_dir() / "logs.db"


def setup_logging() -> None:
    """Configure logging to write to both file and console."""
    log_path = user_dir() / "consortium.log"
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler with ERROR level
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

# Replace existing logging setup with new setup
setup_logging()
logger = logging.getLogger(__name__)
logger.debug("llm_karpathy_consortium module is being imported")

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    
    def __init__(self):
        self.db = sqlite_utils.Database(logs_db_path())
    
    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        """Get singleton database connection."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.db

def log_response(response, model):
    """Log model response to database and log file."""
    try:
        db = DatabaseConnection.get_connection()
        response.log_to_db(db)
        logger.debug(f"Response from {model} logged to database")
    except Exception as e:
        logger.error(f"Error logging to database: {e}")

class IterationContext:
    def __init__(self, synthesis: Dict[str, Any], model_responses: List[Dict[str, Any]]):
        self.synthesis = synthesis
        self.model_responses = model_responses

class ConsortiumOrchestrator:
    def __init__(
        self,
        models: List[str],
        system_prompt: Optional[str] = None,
        confidence_threshold: float = 0.8,
        max_iterations: int = 3,
        arbiter: Optional[str] = None,
    ):
        self.models = models
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.arbiter = arbiter or "claude-3-opus-20240229"
        self.iteration_history: List[IterationContext] = []

    async def orchestrate(self, prompt: str) -> Dict[str, Any]:
        iteration_count = 0
        final_result = None
        original_prompt = prompt
        current_prompt = f"""<prompt>
    <instruction>{prompt}</instruction>
</prompt>"""

        while iteration_count < self.max_iterations:
            iteration_count += 1
            logger.debug(f"Starting iteration {iteration_count}")

            # Get responses from all models using the current prompt
            model_responses = await self._get_model_responses(current_prompt)

            # Have arbiter synthesize and evaluate responses
            synthesis = await self._synthesize_responses(original_prompt, model_responses)

            # Store iteration context
            self.iteration_history.append(IterationContext(synthesis, model_responses))

            if synthesis["confidence"] >= self.confidence_threshold:
                final_result = synthesis
                break

            # Prepare for next iteration if needed
            current_prompt = self._construct_iteration_prompt(original_prompt, synthesis)

        if final_result is None:
            final_result = synthesis

        return {
            "original_prompt": original_prompt,
            "model_responses": model_responses,
            "synthesis": final_result,
            "metadata": {
                "models_used": self.models,
                "arbiter": self.arbiter,
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_count": iteration_count
            }
        }

    async def _get_model_responses(self, prompt: str) -> List[Dict[str, Any]]:
        tasks = [self._get_model_response(model, prompt) for model in self.models]
        return await asyncio.gather(*tasks)

    async def _get_model_response(self, model: str, prompt: str) -> Dict[str, Any]:
        logger.debug(f"Getting response from model: {model}")
        try:
            xml_prompt = f"""<prompt>
    <instruction>{prompt}</instruction>
</prompt>"""
            response = llm.get_model(model).prompt(xml_prompt, system=self.system_prompt)
            log_response(response, model)
            return {
                "model": model,
                "response": response.text(),
                "confidence": self._extract_confidence(response.text()),
            }
        except Exception as e:
            logger.exception(f"Error getting response from {model}")
            return {"model": model, "error": str(e)}

    def _parse_confidence_value(self, text: str, default: float = 0.5) -> float:
        """Helper method to parse confidence values consistently."""
        # Try to find XML confidence tag
        xml_match = re.search(r"<confidence>\s*(0?\.\d+|1\.0|\d+)\s*</confidence>", text, re.DOTALL)
        if xml_match:
            try:
                value = float(xml_match.group(1))
                return value / 100 if value > 1 else value
            except ValueError:
                pass
        
        # Fallback to plain text parsing
        for line in text.lower().split("\n"):
            if "confidence:" in line or "confidence level:" in line:
                try:
                    nums = re.findall(r"(\d*\.?\d+)%?", line)
                    if nums:
                        num = float(nums[0])
                        return num / 100 if num > 1 else num
                except (IndexError, ValueError):
                    pass
        
        return default

    def _extract_confidence(self, text: str) -> float:
        return self._parse_confidence_value(text)

    def _construct_iteration_prompt(self, original_prompt: str, last_synthesis: Dict[str, Any]) -> str:
        # Todo: This does not apear in prompt logs db. Either it is not being called or it is not being logged.
        return f"""You are part of a model consortium working together to solve complex problems through an iterative process. Your task is to provide an updated response to a problem, considering previous work and focusing on specific refinement areas.

Review the previous iterations of work on this problem:

<previous_iterations>
{self._format_iteration_history()}
</previous_iterations>

Consider the most recent synthesis and the areas identified for refinement:

<previous_synthesis>
{last_synthesis['synthesis']}
</previous_synthesis>

<refinement_areas>
{self._format_refinement_areas(last_synthesis['refinement_areas'])}
</refinement_areas>

Here's the original prompt you're addressing:

<original_prompt>
{original_prompt}
</original_prompt>

Instructions:
1. Analyze the original prompt, previous iterations, and refinement areas.
2. Provide an updated response that addresses the refinement areas while considering the full context.
3. Explain your reasoning thoroughly.
4. Include a confidence level (0-1) for your response.

Please structure your response as follows:

<problem_breakdown>
- Summarize key points from previous iterations and synthesis.
- Identify patterns or trends across iterations.
- List potential approaches to address each refinement area.
- Provide a detailed analysis of the problem, previous work, and refinement areas. Break down your thought process and consider different approaches.
</problem_breakdown>

<updated_response>
Provide your updated response, addressing the refinement areas and incorporating insights from your analysis. For each refinement area, explicitly state how it is addressed in your response.
</updated_response>

<reasoning>
Explain your reasoning for the updated response, referencing specific points from your analysis and how they informed your decisions.
</reasoning>

<confidence>
State your confidence level as a number between 0 and 1.
</confidence>

Remember to be thorough in your reasoning and consider all aspects of the problem before providing your final response."""

    def _format_iteration_history(self) -> str:
        history = []
        for i, iteration in enumerate(self.iteration_history, start=1):
            model_responses = "\n".join(
                f"<model_response>{r['model']}: {r.get('response', 'Error')}</model_response>"
                for r in iteration.model_responses
            )
            
            history.append(f"""<iteration>
            <iteration_number>{i}</iteration_number>
            <model_responses>
                {model_responses}
            </model_responses>
            <synthesis>{iteration.synthesis['synthesis']}</synthesis>
            <confidence>{iteration.synthesis['confidence']}</confidence>
            <refinement_areas>
                {self._format_refinement_areas(iteration.synthesis['refinement_areas'])}
            </refinement_areas>
        </iteration>""")
        return "\n".join(history) if history else "<no_previous_iterations>No previous iterations available.</no_previous_iterations>"

    def _format_refinement_areas(self, areas: List[str]) -> str:
        return "\n                ".join(f"<area>{area}</area>" for area in areas)

    def _format_responses(self, responses: List[Dict[str, Any]]) -> str:
        formatted = []
        for r in responses:
            formatted.append(f"""<model_response>
            <model>{r['model']}</model>
            <confidence>{r.get('confidence', 'N/A')}</confidence>
            <response>{r.get('response', 'Error: ' + r.get('error', 'Unknown error'))}</response>
        </model_response>""")
        return "\n".join(formatted)

    async def _synthesize_responses(self, original_prompt: str, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.debug("Synthesizing responses")
        arbiter = llm.get_model(self.arbiter)
        
        formatted_history = self._format_iteration_history()
        formatted_responses = self._format_responses(responses)
        
        arbiter_prompt = f"""You are an advanced AI synthesis system designed to analyze and combine multiple AI-generated responses into a comprehensive and well-reasoned final output. Your task is to review the following information and produce a synthesized response that represents the best consensus while highlighting important dissenting views.

Here is the iteration history of the responses:

<iteration_history>
{formatted_history}
</iteration_history>

Here are the model responses to be analyzed:

<model_responses>
{formatted_responses}
</model_responses>

Here is the original prompt that generated the responses:

<original_prompt>
{original_prompt}
</original_prompt>

Please follow these steps to complete your task:

1. Carefully analyze the original prompt, iteration history, and model responses.
2. Extract and list key points from each model response.
3. Compare and contrast the key points from different responses.
4. Evaluate the relevance of each response to the original prompt.
5. Identify areas of agreement and disagreement among the responses.
6. Synthesize a final response that represents the best consensus.
7. Determine your confidence level in the synthesized response.
8. Highlight any important dissenting views.
9. Assess whether further iterations are needed.
10. If further iterations are needed, provide recommendations for refinement areas.

Wrap your thought process inside <thought_process> tags before providing the final output. In your thought process, consider the following questions:
- What are the key points addressed by each model response?
- How do the responses align or differ from each other?
- What are the strengths and weaknesses of each response?
- Are there any unique insights or perspectives offered by specific responses?
- How well does each response address the original prompt?

After your thought process, provide your synthesized output using the following format:

<synthesis_output>
    <synthesis>
        [Your synthesized response here. This should be a comprehensive summary that combines the best elements of the analyzed responses while addressing the original prompt effectively.]
    </synthesis>
    
    <confidence>
        [Your confidence in this synthesis, expressed as a decimal between 0 and 1. For example, 0.85 would indicate 85% confidence.]
    </confidence>
    
    <analysis>
        [A concise summary of your analysis, explaining how you arrived at your synthesized response and confidence level.]
    </analysis>
    
    <dissent>
        [List any notable dissenting views or alternative perspectives that were not incorporated into the main synthesis but are still worth considering.]
    </dissent>
    
    <needs_iteration>
        [Indicate whether further iteration is needed. Use "true" if more refinement is necessary, or "false" if the current synthesis is sufficient.]
    </needs_iteration>
    
    <refinement_areas>
        [If needs_iteration is true, provide a list of specific areas or aspects that require further refinement or exploration in subsequent iterations.]
    </refinement_areas>
</synthesis_output>

Remember to maintain objectivity and consider all perspectives fairly in your analysis and synthesis. Your goal is to provide a comprehensive and balanced response that accurately represents the collective insights from the model responses while addressing the original prompt effectively."""

        arbiter_response = arbiter.prompt(arbiter_prompt)
        log_response(arbiter_response, arbiter)

        try:
            return self._parse_arbiter_response(arbiter_response.text())
        except Exception as e:
            logger.exception("Error parsing arbiter response")
            return {
                "synthesis": arbiter_response.text(),
                "confidence": 0.5,
                "analysis": "Parsing failed - see raw response",
                "dissent": "",
                "needs_iteration": False,
                "refinement_areas": []
            }

    def _parse_arbiter_response(self, text: str) -> Dict[str, Any]:
        sections = {
            "synthesis": r"<synthesis>([\s\S]*?)</synthesis>",
            "confidence": r"<confidence>\s*(0?\.\d+|1\.0|\d+)\s*</confidence>",
            "analysis": r"<analysis>([\s\S]*?)</analysis>",
            "dissent": r"<dissent>([\s\S]*?)</dissent>",
            "needs_iteration": r"<needs_iteration>(true|false)</needs_iteration>",
            "refinement_areas": r"<refinement_areas>([\s\S]*?)</refinement_areas>"
        }

        result = {}
        for key, pattern in sections.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                if key == "confidence":
                    result[key] = self._parse_confidence_value(match.group(1))
                elif key == "needs_iteration":
                    result[key] = match.group(1).lower() == "true"
                elif key == "refinement_areas":
                    result[key] = [area.strip() for area in match.group(1).split("\n") if area.strip()]
                else:
                    result[key] = match.group(1).strip()
            else:
                result[key] = "" if key != "confidence" else 0.5

        return result

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("prompt")
    @click.option(
        "-m",
        "--models",
        multiple=True,
        help="Models to include in consortium (can specify multiple)",
        default=[
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "gpt-4",
            "gemini-pro"
        ],
    )
    @click.option(
        "--arbiter",
        help="Model to use as arbiter",
        default="claude-3-opus-20240229"
    )
    @click.option(
        "--confidence-threshold",
        type=float,
        help="Minimum confidence threshold",
        default=0.8
    )
    @click.option(
        "--max-iterations",
        type=int,
        help="Maximum number of iteration rounds",
        default=3
    )
    @click.option(
        "--system",
        help="System prompt to use",
    )
    @click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True),
        help="Save full results to this JSON file",
    )
    def consortium(
        prompt: str,
        models: List[str],
        arbiter: str,
        confidence_threshold: float,
        max_iterations: int,
        system: Optional[str],
        output: Optional[str],
    ):
        """Run prompt through a consortium of models and synthesize results."""
        logger.info(f"Starting consortium with {len(models)} models")
        logger.debug(f"Models: {', '.join(models)}")
        logger.debug(f"Arbiter model: {arbiter}")
        
        orchestrator = ConsortiumOrchestrator(
            models=list(models),
            system_prompt=system,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
            arbiter=arbiter,
        )
        
        try:
            result = asyncio.run(orchestrator.orchestrate(prompt))
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {output}")
            
            click.echo("\nSynthesized response:\n")
            click.echo(result["synthesis"]["synthesis"])
            
            click.echo(f"\nConfidence: {result['synthesis']['confidence']}")
            
            click.echo("\nAnalysis:")
            click.echo(result["synthesis"]["analysis"])
            
            if result["synthesis"]["dissent"]:
                click.echo("\nNotable dissenting views:")
                click.echo(result["synthesis"]["dissent"])
            
            click.echo(f"\nNumber of iterations: {result['metadata']['iteration_count']}")
            
            click.echo("\nIndividual model responses:")
            for response in result["model_responses"]:
                click.echo(f"\nModel: {response['model']}")
                click.echo(f"Confidence: {response.get('confidence', 'N/A')}")
                click.echo(f"Response: {response.get('response', 'Error: ' + response.get('error', 'Unknown error'))}")
                
        except Exception as e:
            logger.exception("Error in consortium command")
            raise click.ClickException(str(e))

class KarpathyConsortiumPlugin:
    @staticmethod
    @llm.hookimpl
    def register_commands(cli):
        logger.debug("KarpathyConsortiumPlugin.register_commands called")

logger.debug("llm_karpathy_consortium module finished loading")

# Ensure the KarpathyConsortiumPlugin is exported
__all__ = ['KarpathyConsortiumPlugin', 'log_response', 'DatabaseConnection', 'logs_db_path', 'user_dir']

__version__ = "0.1.0"
