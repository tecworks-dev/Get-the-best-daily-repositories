# agentic-cursorrules

A Python-based practical approach to managing multiple AI agents in large codebases by enforcing strict file-tree partitioning, designed to prevent conflicts and maintain coherence across complex projects. Inspired by [cursor-boost](https://github.com/grp06/cursor-boost).

<img src="https://github.com/user-attachments/assets/4937c3da-fbd6-49b3-9c22-86ae02dabec7" width="60%">

## Core Concept

This tool addresses a critical challenge in AI-assisted development by preventing merge conflicts and maintaining codebase coherence when using AI assistance across different parts of your codebase, accomplishing this through:

1. Partitioning the codebase into logical domains (e.g., frontend, API, database)
2. Generating domain-specific markdown files with explicit file-tree boundaries
3. Providing clear context and access rules for AI assistants through these markdown files

## Installation

```bash
git clone https://github.com/yourusername/agentic-cursorrules.git .agentic-cursorrules
cd .agentic-cursorrules

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up .cursorrules file
cp .cursorrules.example ../.cursorrules
```

Important note: The `.cursorrules` file needs to be in your current working directory where you'll run the agent generator, though if there's already a `.cursorrules` file available in the root folder, it will take precedence.

## Usage

1. Configure your domains in `config.yaml` with clear architectural boundaries:
```yaml
project_title: "agentic-cursorrules"
tree_focus:
  - "app"    # Frontend logic
  - "api"    # Backend services
  - "db"     # Database layer
  - "api/auth/middleware"    # Specific auth middleware subfolder
  - "app/components/forms"   # Just the forms components
```

For example, with this configuration:
- The `app` agent will see all frontend files EXCEPT those in `components/forms`
- The `api` agent will see all backend files EXCEPT those in `auth/middleware`
- Dedicated agents for `api/auth/middleware` and `app/components/forms` will focus solely on their specific subsystems
- The `db` agent maintains access to all database-related files

This separation allows you to have specialized agents working on form components or authentication middleware without interfering with the broader frontend or backend development efforts.

2. Run the generator with optional recurring updates:
```bash
python main.py
# Or for recurring updates every 60 seconds:
python main.py --recurring
```

3. Reference the generated agent files in your development environment:
```
@agentic-cursorrules_agent_app.md  # Frontend-focused agent
@agentic-cursorrules_agent_api.md  # Backend-focused agent
@agentic-cursorrules_agent_db.md   # Database-focused agent
```

## Default Configuration

The tool comes with sensible defaults for web development projects that can be tailored to your specific needs:

```yaml
important_dirs:
  - components
  - pages
  - app

exclude_dirs:
  - node_modules
  - dist
  - build

include_extensions:
  - .py
  - .ts
  - .tsx
```

## How It Works

1. **Codebase Partitioning**
   - Defines clear boundaries through comprehensive YAML configuration
   - Generates separate file-trees for each domain
   - Creates agent-specific markdown files containing base rules and context

2. **Access Control**
   - Each agent receives only its domain-specific file-tree information
   - Explicit instructions to operate within defined boundaries
   - Clear documentation of domain responsibilities

3. **Conflict Prevention**
   - Physical separation through intelligent file-tree partitioning
   - Clear ownership boundaries for each agent
   - Significantly reduced risk of overlapping modifications

## Best Practices

- Maintain a limit of 3-4 concurrent agents for optimal performance and manageability
- Define clear domain boundaries before initiating development
- Implement semantic naming conventions for domains
- Regularly review agent interactions at domain boundaries
- Consider maintaining separate version control branches per domain

## Example Tree agent_{folder} .md file

```
You are an agent that specializes in the __tests__ directory within app of this project. Your expertise and responses should focus specifically on the code and files within this directory structure:

├── components/
│   ├── Component.test.tsx
│   ├── Overview.test.tsx
│   ├── Analysis.test.tsx
├── hooks/
│   └── hookOne.test.tsx
└── lib/
    └── api/
        └── client.test.ts

When providing assistance, only reference and modify files within this directory structure. If you need to work with files outside this structure, list the required files and ask the user for permission first.
```

## Technical Overview

```yaml
Key Features:
- Sophisticated domain-specific agent rulesets
- Physical separation through intelligent file-tree partitioning
- Advanced conflict prevention via explicit boundary definition
- Optimized support for up to 4 concurrent agents
- Flexible domain configuration through YAML
- Comprehensive markdown-based instruction sets
- Contextual file-tree awareness
```

## Stars

[![Star History Chart](https://api.star-history.com/svg?repos=s-smits/agentic-cursorrules&type=Date)](https://star-history.com/#s-smits/agentic-cursorrules&Date)
