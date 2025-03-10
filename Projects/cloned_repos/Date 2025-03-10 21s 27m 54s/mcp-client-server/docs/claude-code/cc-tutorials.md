Claude Code tutorials - Anthropic

[Anthropic home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/dark.svg)](/)

English

Search...

* [Research](https://www.anthropic.com/research)
* [News](https://www.anthropic.com/news)
* [Go to claude.ai](https://claude.ai/)
* [Go to claude.ai](https://claude.ai/)

Search...

Navigation

Claude Code

Claude Code tutorials

[Welcome](/en/home)[User Guides](/en/docs/welcome)[API Reference](/en/api/getting-started)[Prompt Library](/en/prompt-library/library)[Release Notes](/en/release-notes/overview)[Developer Newsletter](/en/developer-newsletter/overview)

- [Developer Console](https://console.anthropic.com/)
- [Developer Discord](https://www.anthropic.com/discord)
- [Support](https://support.anthropic.com/)

##### Get started

* [Overview](/en/docs/welcome)
* [Initial setup](/en/docs/initial-setup)
* [Intro to Claude](/en/docs/intro-to-claude)

##### Learn about Claude

* Use cases
* Models & pricing
* [Security and compliance](https://trust.anthropic.com/)

##### Build with Claude

* [Define success criteria](/en/docs/build-with-claude/define-success)
* [Develop test cases](/en/docs/build-with-claude/develop-tests)
* [Context windows](/en/docs/build-with-claude/context-windows)
* [Vision](/en/docs/build-with-claude/vision)
* Prompt engineering
* [Extended thinking](/en/docs/build-with-claude/extended-thinking)
* [Multilingual support](/en/docs/build-with-claude/multilingual-support)
* Tool use (function calling)
* [Prompt caching](/en/docs/build-with-claude/prompt-caching)
* [PDF support](/en/docs/build-with-claude/pdf-support)
* [Citations](/en/docs/build-with-claude/citations)
* [Token counting](/en/docs/build-with-claude/token-counting)
* [Batch processing](/en/docs/build-with-claude/batch-processing)
* [Embeddings](/en/docs/build-with-claude/embeddings)

##### Agents and tools

* Claude Code

  + [Overview](/en/docs/agents-and-tools/claude-code/overview)
  + [Claude Code tutorials](/en/docs/agents-and-tools/claude-code/tutorials)
  + [Troubleshooting](/en/docs/agents-and-tools/claude-code/troubleshooting)
* [Computer use (beta)](/en/docs/agents-and-tools/computer-use)
* [Model Context Protocol (MCP)](/en/docs/agents-and-tools/mcp)
* [Google Sheets add-on](/en/docs/agents-and-tools/claude-for-sheets)

##### Test and evaluate

* Strengthen guardrails
* [Using the Evaluation Tool](/en/docs/test-and-evaluate/eval-tool)

##### Administration

* [Admin API](/en/docs/administration/administration-api)

##### Resources

* [Glossary](/en/docs/resources/glossary)
* [Model deprecations](/en/docs/resources/model-deprecations)
* [System status](https://status.anthropic.com/)
* [Claude 3 model card](https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf)
* [Claude 3.7 system card](https://anthropic.com/claude-3-7-sonnet-system-card)
* [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)
* [Anthropic Courses](https://github.com/anthropics/courses)

##### Legal center

* [Anthropic Privacy Policy](https://www.anthropic.com/legal/privacy)

Claude Code

# Claude Code tutorials

Practical examples and patterns for effectively using Claude Code in your development workflow.

This guide provides step-by-step tutorials for common workflows with Claude Code. Each tutorial includes clear instructions, example commands, and best practices to help you get the most from Claude Code.

## [​](#table-of-contents) Table of contents

* [Understand new codebases](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#understand-new-codebases)
* [Fix bugs efficiently](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#fix-bugs-efficiently)
* [Refactor code](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#refactor-code)
* [Work with tests](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#work-with-tests)
* [Create pull requests](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#create-pull-requests)
* [Handle documentation](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#handle-documentation)
* [Use advanced git workflows](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#use-advanced-git-workflows)
* [Work with images](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#work-with-images)
* [Set up project memory](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-project-memory)
* [Use Claude as a unix-style utility](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#use-claude-as-a-unix-style-utility)
* [Create custom slash commands](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#create-custom-slash-commands)
* [Set up Model Context Protocol (MCP)](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp)

## [​](#understand-new-codebases) Understand new codebases

### [​](#get-a-quick-codebase-overview) Get a quick codebase overview

**When to use:** You’ve just joined a new project and need to understand its structure quickly.

1

Navigate to the project root directory

```
$ cd /path/to/project

```

2

Start Claude Code

```
$ claude

```

3

Ask for a high-level overview

```
> give me an overview of this codebase

```

4

Dive deeper into specific components

```
> explain the main architecture patterns used here
> what are the key data models?
> how is authentication handled?

```

**Tips:**

* Start with broad questions, then narrow down to specific areas
* Ask about coding conventions and patterns used in the project
* Request a glossary of project-specific terms

### [​](#find-relevant-code) Find relevant code

**When to use:** You need to locate code related to a specific feature or functionality.

1

Ask Claude to find relevant files

```
> find the files that handle user authentication

```

2

Get context on how components interact

```
> how do these authentication files work together?

```

3

Understand the execution flow

```
> trace the login process from front-end to database

```

**Tips:**

* Be specific about what you’re looking for
* Use domain language from the project

---

## [​](#fix-bugs-efficiently) Fix bugs efficiently

### [​](#diagnose-error-messages) Diagnose error messages

**When to use:** You’ve encountered an error message and need to find and fix its source.

1

Share the error with Claude

```
> I'm seeing an error when I run npm test

```

2

Ask for fix recommendations

```
> suggest a few ways to fix the @ts-ignore in user.ts

```

3

Apply the fix

```
> update user.ts to add the null check you suggested

```

**Tips:**

* Tell Claude the command to reproduce the issue and get a stack trace
* Mention any steps to reproduce the error
* Let Claude know if the error is intermittent or consistent

---

## [​](#refactor-code) Refactor code

### [​](#modernize-legacy-code) Modernize legacy code

**When to use:** You need to update old code to use modern patterns and practices.

1

Identify legacy code for refactoring

```
> find deprecated API usage in our codebase

```

2

Get refactoring recommendations

```
> suggest how to refactor utils.js to use modern JavaScript features

```

3

Apply the changes safely

```
> refactor utils.js to use ES2024 features while maintaining the same behavior

```

4

Verify the refactoring

```
> run tests for the refactored code

```

**Tips:**

* Ask Claude to explain the benefits of the modern approach
* Request that changes maintain backward compatibility when needed
* Do refactoring in small, testable increments

---

## [​](#work-with-tests) Work with tests

### [​](#add-test-coverage) Add test coverage

**When to use:** You need to add tests for uncovered code.

1

Identify untested code

```
> find functions in NotificationsService.swift that are not covered by tests

```

2

Generate test scaffolding

```
> add tests for the notification service

```

3

Add meaningful test cases

```
> add test cases for edge conditions in the notification service

```

4

Run and verify tests

```
> run the new tests and fix any failures

```

**Tips:**

* Ask for tests that cover edge cases and error conditions
* Request both unit and integration tests when appropriate
* Have Claude explain the testing strategy

---

## [​](#create-pull-requests) Create pull requests

### [​](#generate-comprehensive-prs) Generate comprehensive PRs

**When to use:** You need to create a well-documented pull request for your changes.

1

Summarize your changes

```
> summarize the changes I've made to the authentication module

```

2

Generate a PR with Claude

```
> create a pr

```

3

Review and refine

```
> enhance the PR description with more context about the security improvements

```

4

Add testing details

```
> add information about how these changes were tested

```

**Tips:**

* Ask Claude directly to make a PR for you
* Review Claude’s generated PR before submitting
* Ask Claude to highlight potential risks or considerations

## [​](#handle-documentation) Handle documentation

### [​](#generate-code-documentation) Generate code documentation

**When to use:** You need to add or update documentation for your code.

1

Identify undocumented code

```
> find functions without proper JSDoc comments in the auth module

```

2

Generate documentation

```
> add JSDoc comments to the undocumented functions in auth.js

```

3

Review and enhance

```
> improve the generated documentation with more context and examples

```

4

Verify documentation

```
> check if the documentation follows our project standards

```

**Tips:**

* Specify the documentation style you want (JSDoc, docstrings, etc.)
* Ask for examples in the documentation
* Request documentation for public APIs, interfaces, and complex logic

## [​](#work-with-images) Work with images

### [​](#analyze-images-and-screenshots) Analyze images and screenshots

**When to use:** You need to work with images in your codebase or get Claude’s help analyzing image content.

1

Add an image to the conversation

You can use any of these methods:

```
# 1. Drag and drop an image into the Claude Code window

# 2. Copy an image and paste it into the CLI with ctrl+v

# 3. Provide an image path
$ claude
> Analyze this image: /path/to/your/image.png

```

2

Ask Claude to analyze the image

```
> What does this image show?
> Describe the UI elements in this screenshot
> Are there any problematic elements in this diagram?

```

3

Use images for context

```
> Here's a screenshot of the error. What's causing it?
> This is our current database schema. How should we modify it for the new feature?

```

4

Get code suggestions from visual content

```
> Generate CSS to match this design mockup
> What HTML structure would recreate this component?

```

**Tips:**

* Use images when text descriptions would be unclear or cumbersome
* Include screenshots of errors, UI designs, or diagrams for better context
* You can work with multiple images in a conversation
* Image analysis works with diagrams, screenshots, mockups, and more

---

## [​](#set-up-project-memory) Set up project memory

### [​](#create-an-effective-claude-md-file) Create an effective CLAUDE.md file

**When to use:** You want to set up a CLAUDE.md file to store important project information, conventions, and frequently used commands.

1

Bootstrap a CLAUDE.md for your codebase

```
> /init

```

**Tips:**

* Include frequently used commands (build, test, lint) to avoid repeated searches
* Document code style preferences and naming conventions
* Add important architectural patterns specific to your project
* You can add CLAUDE.md files to any of:
  + The folder you run Claude in: Automatically added to conversations you start in that folder
  + Child directories: Claude pulls these in on demand
  + *~/.claude/CLAUDE.md*: User-specific preferences that you don’t want to check into source control

---

## [​](#use-claude-as-a-unix-style-utility) Use Claude as a unix-style utility

### [​](#add-claude-to-your-verification-process) Add Claude to your verification process

**When to use:** You want to use Claude Code as a linter or code reviewer.

**Steps:**

1

Add Claude to your build script

```
// package.json
{
    ...
    "scripts": {
        ...
        "lint:claude": "claude -p 'you are a linter. please look at the changes vs. main and report any issues related to typos. report the filename and line number on one line, and a description of the issue on the second line. do not return any other text.'"
    }
}

```

### [​](#pipe-in-pipe-out) Pipe in, pipe out

**When to use:** You want to pipe data into Claude, and get back data in a structured format.

1

Pipe data through Claude

```
$ cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt

```

---

## [​](#create-custom-slash-commands) Create custom slash commands

Claude Code supports custom slash commands that you can create to quickly execute specific prompts or tasks.

### [​](#create-project-specific-commands) Create project-specific commands

**When to use:** You want to create reusable slash commands for your project that all team members can use.

1

Create a commands directory in your project

```
$ mkdir -p .claude/commands

```

2

Create a Markdown file for each command

```
$ echo "Analyze the performance of this code and suggest three specific optimizations:" > .claude/commands/optimize.md

```

3

Use your custom command in Claude Code

```
$ claude
> /project:optimize

```

**Tips:**

* Command names are derived from the filename (e.g., `optimize.md` becomes `/project:optimize`)
* You can organize commands in subdirectories (e.g., `.claude/commands/frontend/component.md` becomes `/project:frontend:component`)
* Project commands are available to everyone who clones the repository
* The Markdown file content becomes the prompt sent to Claude when the command is invoked

### [​](#create-personal-slash-commands) Create personal slash commands

**When to use:** You want to create personal slash commands that work across all your projects.

1

Create a commands directory in your home folder

```
$ mkdir -p ~/.claude/commands

```

2

Create a Markdown file for each command

```
$ echo "Review this code for security vulnerabilities, focusing on:" > ~/.claude/commands/security-review.md

```

3

Use your personal custom command

```
$ claude
> /user:security-review

```

**Tips:**

* Personal commands are prefixed with `/user:` instead of `/project:`
* Personal commands are only available to you and not shared with your team
* Personal commands work across all your projects
* You can use these for consistent workflows across different codebases

## [​](#set-up-model-context-protocol-mcp) Set up Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open protocol that enables LLMs to access external tools and data sources. For more details, see the [MCP documentation](https://modelcontextprotocol.io/introduction).

Use third party MCP servers at your own risk. Make sure you trust the MCP
servers, and be especially careful when using MCP servers that talk to the
internet, as these can expose you to prompt injection risk.

### [​](#configure-mcp-servers) Configure MCP servers

**When to use:** You want to enhance Claude’s capabilities by connecting it to specialied tools and external servers using the Model Context Protocol.

1

Add an MCP Stdio Server

```
# Basic syntax
$ claude mcp add <name> <command> [args...]

# Example: Adding a local server
$ claude mcp add my-server -e API_KEY=123 -- /path/to/server arg1 arg2

```

2

Manage your MCP servers

```
# List all configured servers
$ claude mcp list

# Get details for a specific server
$ claude mcp get my-server

# Remove a server
$ claude mcp remove my-server

```

**Tips:**

* Use the `-s` or `--scope` flag with `project` (default) or `global` to specify where the configuration is stored
* Set environment variables with `-e` or `--env` flags (e.g., `-e KEY=value`)
* MCP follows a client-server architecture where Claude Code (the client) can connect to multiple specialized servers

### [​](#connect-to-a-postgres-mcp-server) Connect to a Postgres MCP server

**When to use:** You want to give Claude read-only access to a PostgreSQL database for querying and schema inspection.

1

Add the Postgres MCP server

```
$ claude mcp add postgres-server /path/to/postgres-mcp-server --connection-string "postgresql://user:pass@localhost:5432/mydb"

```

2

Query your database with Claude

```
# In your Claude session, you can ask about your database
$ claude
> describe the schema of our users table
> what are the most recent orders in the system?
> show me the relationship between customers and invoices

```

**Tips:**

* The Postgres MCP server provides read-only access for safety
* Claude can help you explore database structure and run analytical queries
* You can use this to quickly understand database schemas in unfamiliar projects
* Make sure your connection string uses appropriate credentials with minimum required permissions

---

## [​](#next-steps) Next steps

[## Claude Code reference implementation

Clone our development container reference implementation.](https://github.com/anthropics/claude-code/tree/main/.devcontainer)

Was this page helpful?

YesNo

[Overview](/en/docs/agents-and-tools/claude-code/overview)[Troubleshooting](/en/docs/agents-and-tools/claude-code/troubleshooting)

[x](https://x.com/AnthropicAI)[linkedin](https://www.linkedin.com/company/anthropicresearch)

On this page

* [Table of contents](#table-of-contents)
* [Understand new codebases](#understand-new-codebases)
* [Get a quick codebase overview](#get-a-quick-codebase-overview)
* [Find relevant code](#find-relevant-code)
* [Fix bugs efficiently](#fix-bugs-efficiently)
* [Diagnose error messages](#diagnose-error-messages)
* [Refactor code](#refactor-code)
* [Modernize legacy code](#modernize-legacy-code)
* [Work with tests](#work-with-tests)
* [Add test coverage](#add-test-coverage)
* [Create pull requests](#create-pull-requests)
* [Generate comprehensive PRs](#generate-comprehensive-prs)
* [Handle documentation](#handle-documentation)
* [Generate code documentation](#generate-code-documentation)
* [Work with images](#work-with-images)
* [Analyze images and screenshots](#analyze-images-and-screenshots)
* [Set up project memory](#set-up-project-memory)
* [Create an effective CLAUDE.md file](#create-an-effective-claude-md-file)
* [Use Claude as a unix-style utility](#use-claude-as-a-unix-style-utility)
* [Add Claude to your verification process](#add-claude-to-your-verification-process)
* [Pipe in, pipe out](#pipe-in-pipe-out)
* [Create custom slash commands](#create-custom-slash-commands)
* [Create project-specific commands](#create-project-specific-commands)
* [Create personal slash commands](#create-personal-slash-commands)
* [Set up Model Context Protocol (MCP)](#set-up-model-context-protocol-mcp)
* [Configure MCP servers](#configure-mcp-servers)
* [Connect to a Postgres MCP server](#connect-to-a-postgres-mcp-server)
* [Next steps](#next-steps)