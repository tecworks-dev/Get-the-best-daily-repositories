"""
Advanced usage example demonstrating custom configuration and handling of the LLM Karpathy Consortium.
"""
import asyncio
import logging
from llm_consortium import ConsortiumOrchestrator, setup_logging

# Custom system prompt example
CUSTOM_SYSTEM_PROMPT = """You are part of an expert panel tasked with providing detailed technical analysis.
Please ensure your responses include:
1. Technical accuracy
2. Practical implications
3. Current state of research
4. Potential future developments

Structure your response as follows:
<technical_analysis>[Your analysis]</technical_analysis>
<practical_implications>[Implications]</practical_implications>
<confidence>[0-1 value]</confidence>
"""

async def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize orchestrator with custom settings
    orchestrator = ConsortiumOrchestrator(
        models=[
            "claude-3-opus-20240229",
            "gpt-4",
            "gemini-pro"
        ],
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        confidence_threshold=0.9,
        max_iterations=5,
        arbiter_model="claude-3-opus-20240229"
    )
    
    try:
        # Complex technical prompt
        prompt = """
        Analyze the potential impact of quantum computing on current cryptographic systems,
        considering:
        1. Timeline for quantum supremacy
        2. Vulnerable encryption methods
        3. Post-quantum cryptography solutions
        4. Implementation challenges
        """
        
        # Run consortium with error handling
        result = await orchestrator.orchestrate(prompt)
        
        # Process and display results
        print("\nFinal Synthesis:")
        print(result['synthesis']['synthesis'])
        
        print(f"\nConfidence: {result['synthesis']['confidence']}")
        
        print("\nModel Responses:")
        for response in result['model_responses']:
            print(f"\nModel: {response['model']}")
            print(f"Confidence: {response.get('confidence', 'N/A')}")
            
        # Log results
        logger.info(f"Successfully processed prompt with {result['metadata']['iteration_count']} iterations")
        
    except Exception as e:
        logger.error(f"Error during consortium execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
