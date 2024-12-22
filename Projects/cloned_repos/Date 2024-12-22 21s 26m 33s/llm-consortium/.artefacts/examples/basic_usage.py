"""
Basic usage example of the LLM Karpathy Consortium.
"""
import asyncio
from llm_consortium import ConsortiumOrchestrator

async def main():
    # Initialize the orchestrator with default settings
    orchestrator = ConsortiumOrchestrator(
        models=[
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "gpt-4",
            "gemini-pro"
        ]
    )

    # Simple prompt example
    prompt = "Explain the concept of quantum entanglement to a high school student"
    
    # Run the consortium
    result = await orchestrator.orchestrate(prompt)
    
    # Print results
    print("\nSynthesized Response:")
    print(result['synthesis']['synthesis'])
    
    print("\nConfidence Level:", result['synthesis']['confidence'])
    
    print("\nAnalysis:")
    print(result['synthesis']['analysis'])
    
    if result['synthesis']['dissent']:
        print("\nDissenting Views:")
        print(result['synthesis']['dissent'])
    
    print("\nNumber of Iterations:", result['metadata']['iteration_count'])

if __name__ == "__main__":
    asyncio.run(main())
