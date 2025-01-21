import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with DeepSeek base URL
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Verify API key is present
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set. Please check your .env file.")

def format_latex(text):
    """Format inline LaTeX expressions for proper rendering in Gradio."""
    # Replace single dollar signs with double for better display
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Skip lines that already have double dollars
        if '$$' in line:
            formatted_lines.append(line)
            continue
            
        # Format single dollar expressions
        in_math = False
        new_line = ''
        for i, char in enumerate(line):
            if char == '$' and (i == 0 or line[i-1] != '\\'):
                in_math = not in_math
                new_line += '$$' if in_math else '$$'
            else:
                new_line += char
        formatted_lines.append(new_line)
    
    return '\n'.join(formatted_lines)

def chat_with_deepseek(message, history):
    # Convert history to the format expected by the API
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    
    # Call the DeepSeek API
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        
        # Get both reasoning and final content
        reasoning = format_latex(response.choices[0].message.reasoning_content)
        answer = format_latex(response.choices[0].message.content)
        
        # Return both, separated by a clear delimiter
        return f"🤔 Reasoning:\n{reasoning}\n\n📝 Answer:\n{answer}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface with markdown enabled
iface = gr.ChatInterface(
    chat_with_deepseek,
    title="DeepSeek Reasoning Chat",
    description="Chat with DeepSeek's Reasoning model. The model will show its reasoning process before providing the final answer. Supports LaTeX math expressions using $ or $$.",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch() 