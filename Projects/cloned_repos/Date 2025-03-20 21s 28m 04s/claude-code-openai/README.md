# Claude Code but with OpenAI Models 🧙‍♂️🔄 ¯\\_(ツ)_/¯

**Use Claude Code with OpenAI Models** 🤝

A proxy server that lets you use Claude Code with OpenAI models like GPT-4o / gpt-4.5 and o3-mini. 🌉


![Claude Code but with OpenAI Models](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install UV**:
   ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Configure your API keys**:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your-openai-key
   # Optional: customize which models are used
   # BIG_MODEL=gpt-4o
   # SMALL_MODEL=gpt-4o-mini
   ```

4. **Start the proxy server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082
   ```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use OpenAI models through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to OpenAI models:

| Claude Model | OpenAI Model |
|--------------|--------------|
| haiku | gpt-4o-mini (default) |
| sonnet | gpt-4o (default) |

### Customizing Model Mapping

You can customize which OpenAI models are used via environment variables:

- `BIG_MODEL`: The OpenAI model to use for Claude Sonnet models (default: "gpt-4o")
- `SMALL_MODEL`: The OpenAI model to use for Claude Haiku models (default: "gpt-4o-mini")

Add these to your `.env` file to customize:
```
OPENAI_API_KEY=your-openai-key
BIG_MODEL=gpt-4o
SMALL_MODEL=gpt-4o-mini
```

Or set them directly when running the server:
```bash
BIG_MODEL=gpt-4o SMALL_MODEL=gpt-4o-mini uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to OpenAI format via LiteLLM 🔄
3. **Sending** the translated request to OpenAI 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
