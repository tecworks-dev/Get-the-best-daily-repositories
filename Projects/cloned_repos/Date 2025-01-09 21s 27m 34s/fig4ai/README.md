# fig4ai

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A CLI tool that uses AI to generate design rules and documentation from your Figma files. It analyzes your Figma designs and automatically extracts design tokens, components, and layout information into a structured format.

## Overview



https://github.com/user-attachments/assets/c80b7eee-7027-4872-ae30-5279289ff6f7



## Features

- 🎨 Extract design tokens (colors, typography, spacing, effects)
- 🧩 Generate component documentation
- 📐 Analyze layout structures
- 🤖 AI-powered pseudo-code generation
- 🔄 Real-time progress indicators
- 📝 Markdown output format

## Run
Run directly with npx:

```bash
npx fig4ai <figma-url> [--model=claude|gpt4] [--no-ai]
```

## IDE Integration

After generating your `.designrules` file, you can use it with AI-powered IDEs to automatically generate code and configurations:

### Cursor, Windsurf, VS Code

Simply mention the `.designrules` file in your prompts:

```
> Generate a Tailwind config based on @.designrules file
```
```
> Create a Vue login page using the design tokens from @.designrules
```
```
> Build a React component library following @.designrules specifications
```


The AI will analyze your `.designrules` file and generate code that matches your design system's:
- Color palette
- Typography scales
- Spacing system
- Component structures
- Layout patterns
- Shadow effects
- Border styles
- And more...

## How it Works

fig4ai follows a sophisticated process to transform your Figma designs into AI-ready context:

1. **Data Extraction**
   - Connects to Figma API and retrieves comprehensive file data
   - Processes complex nested JSON structure containing all design information

2. **Design Token Parsing**
   - Parses the JSON structure hierarchically: Canvas > Frame > Component / Instance
   - Extracts design tokens (colors, typography, spacing, effects)
   - Organizes components and their instances with style references
   - Maintains relationship between components and their variants

3. **AI-Powered Transformation**
   - For each Canvas, sends structured data to GPT-4o
   - Generates semantic pseudo-code with complete styling context
   - Preserves all design decisions, constraints, and relationships
   - Includes accessibility considerations and responsive behaviors

4. **Structured Documentation**
   - Stores all design tokens and pseudo-code representations in `.designrules`
   - Uses Markdown format for maximum compatibility
   - Maintains hierarchical structure of the design system
   - Preserves all style references and component relationships

5. **AI Context Integration**
   - `.designrules` file serves as a comprehensive design context
   - When mentioned in AI-powered IDEs (Cursor/Windsurf), the file is parsed
   - AI understands the complete design system and can generate accurate code
   - Enables context-aware code generation based on your design system

In essence, fig4ai transforms your Figma file into a structured AI context, making your design system programmatically accessible to AI tools.

## Usage

### Command Line

```bash
npx fig4ai <figma-url> [--model=claude|gpt4] [--no-ai]
```

Or if you've set `FIGMA_DESIGN_URL` in your `.env` file:

```bash
npx fig4ai [--model=claude|gpt4] [--no-ai]
```

### AI Options

The tool supports two AI models for enhanced design analysis:

1. **Claude (Default)**
   - Uses Anthropic's Claude 3 Sonnet model
   - Set `CLAUDE_API_KEY` in your environment variables
   - Generally better at understanding design context
   - More detailed component analysis

2. **GPT-4o**
   - Uses OpenAI's GPT-4 model
   - Set `OPENAI_API_KEY` in your environment variables
   - Alternative option if you prefer OpenAI

You can also run without AI enhancement:
```bash
npx fig4ai <figma-url> --no-ai
```
This will output raw design data in a structured format without AI processing.

### Environment Setup

```env
# Required
FIGMA_ACCESS_TOKEN=your_figma_token

# Optional - At least one needed for AI features
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional
FIGMA_DESIGN_URL=your_default_figma_url
```

### Output

The tool generates a `.designrules` file containing:

- Design token documentation
- Component specifications
- Layout structures
- AI-generated pseudo-code
- Style references
- Accessibility considerations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please:
1. Check the [issues page](https://github.com/f/fig4ai/issues)
2. Create a new issue if your problem isn't already listed
