# LLM Karpathy Consortium API Reference

## ConsortiumOrchestrator

The main class coordinating the model consortium functionality.

### Constructor

```python
ConsortiumOrchestrator(
    models: List[str],
    system_prompt: Optional[str] = None,
    confidence_threshold: float = 0.8,
    max_iterations: int = 3,
    arbiter_model: Optional[str] = None
)
```

#### Parameters:
- `models`: List of model identifiers to include in the consortium
- `system_prompt`: Optional custom system prompt (defaults to DEFAULT_SYSTEM_PROMPT)
- `confidence_threshold`: Minimum confidence level required (0.0-1.0)
- `max_iterations`: Maximum number of refinement iterations
- `arbiter_model`: Model to use for response synthesis (defaults to "claude-3-opus-20240229")

### Methods

#### orchestrate

```python
async def orchestrate(self, prompt: str) -> Dict[str, Any]
```

Orchestrates the consortium process for a given prompt.

**Parameters:**
- `prompt`: The input prompt to process

**Returns:**
Dictionary containing:
- `original_prompt`: The initial prompt
- `model_responses`: List of individual model responses
- `synthesis`: Final synthesized response
- `metadata`: Execution metadata

### Database Functions

#### DatabaseConnection

Singleton class managing database connections.

```python
class DatabaseConnection:
    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        """Get singleton database connection."""
```

#### log_response

```python
def log_response(response, model)
```

Logs model responses to the database.

### Utility Functions

#### setup_logging

```python
def setup_logging() -> None
```

Configures logging for console and file output.

#### user_dir

```python
def user_dir() -> pathlib.Path
```

Returns the user directory for storing application data.

#### logs_db_path

```python
def logs_db_path() -> pathlib.Path
```

Returns the path to the logs database.

## Response Format

### Model Response Structure
```xml
<thought_process>
[Detailed reasoning]
</thought_process>

<answer>
[Final answer]
</answer>

<confidence>
[Confidence level (0-1)]
</confidence>
```

### Synthesis Response Structure
```xml
<synthesis_output>
    <synthesis>[Synthesized response]</synthesis>
    <confidence>[Confidence level]</confidence>
    <analysis>[Analysis details]</analysis>
    <dissent>[Dissenting views]</dissent>
    <needs_iteration>[true/false]</needs_iteration>
    <refinement_areas>[Areas needing refinement]</refinement_areas>
</synthesis_output>
```

## CLI Interface

The plugin adds the `consortium` command to the LLM CLI:

```bash
llm consortium [OPTIONS] PROMPT
```

### Options
- `-m, --models`: Models to include (multiple)
- `--arbiter-model`: Arbiter model
- `--confidence-threshold`: Minimum confidence
- `--max-iterations`: Maximum iterations
- `--system`: Custom system prompt
- `--output`: JSON output file path

## Environment Variables

- `LLM_USER_PATH`: Custom path for application data
