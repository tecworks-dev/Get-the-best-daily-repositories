Claude Code overview - Anthropic

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

Claude Code overview

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

# Claude Code overview

Learn about Claude Code, an agentic coding tool made by Anthropic. Currently in beta as a research preview.

```
npm install -g @anthropic-ai/claude-code

```

Do NOT use `sudo npm install -g` as this can lead to permission issues and security risks. If you encounter permission errors, see the [configuration section](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#configure-claude-code) for recommended solutions.

Claude Code is an agentic coding tool that lives in your terminal, understands your codebase, and helps you code faster through natural language commands. By integrating directly with your development environment, Claude Code streamlines your workflow without requiring additional servers or complex setup.

Claude Code’s key capabilities include:

* Editing files and fixing bugs across your codebase
* Answering questions about your code’s architecture and logic
* Executing and fixing tests, linting, and other commands
* Searching through git history, resolving merge conflicts, and creating commits and PRs

**Research preview**

Code is in beta as a research preview. We’re gathering developer feedback on AI collaboration preferences, which workflows benefit most from AI assistance, and how to improve the agent experience.

This early version will evolve based on user feedback. We plan to enhance tool execution reliability, support for long-running commands, terminal rendering, and Claude’s self-knowledge of its capabilities in the coming weeks.

Report bugs directly with the `/bug` command or through our [GitHub repository](https://github.com/anthropics/claude-code).

---

## [​](#before-you-begin) Before you begin

### [​](#check-system-requirements) Check system requirements

* **Operating Systems**: macOS 10.15+, Ubuntu 20.04+/Debian 10+, or Windows via WSL
* **Hardware**: 4GB RAM minimum
* **Software**:
  + Node.js 18+
  + [git](https://git-scm.com/downloads) 2.23+ (optional)
  + [GitHub](https://cli.github.com/) or [GitLab](https://gitlab.com/gitlab-org/cli) CLI for PR workflows (optional)
  + [ripgrep](https://github.com/BurntSushi/ripgrep?tab=readme-ov-file#installation) (rg) for enhanced file search (optional)
* **Network**: Internet connection required for authentication and AI processing
* **Location**: Available only in [supported countries](https://www.anthropic.com/supported-countries)

**Troubleshooting WSL installation**

Currently, Claude Code does not run directly in Windows, and instead requires WSL. If you encounter issues in WSL:

1. **OS/platform detection issues**: If you receive an error during installation, WSL may be using Windows `npm`. Try:

   * Run `npm config set os linux` before installation
   * Install with `npm install -g @anthropic-ai/claude-code --force --no-os-check` (Do NOT use `sudo`)
2. **Node not found errors**: If you see `exec: node: not found` when running `claude`, your WSL environment may be using a Windows installation of Node.js. You can confirm this with `which npm` and `which node`, which should point to Linux paths starting with `/usr/` rather than `/mnt/c/`. To fix this, try installing Node via your Linux distribution’s package manager or via [`nvm`](https://github.com/nvm-sh/nvm).

### [​](#install-and-authenticate) Install and authenticate

1

Install Claude Code

Run in your terminal: `npm install -g @anthropic-ai/claude-code`

Do NOT use `sudo npm install -g` as this can lead to permission issues and security risks. If you encounter permission errors, see the [configuration section](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#configure-claude-code) for recommended solutions.

2

Navigate to your project

`cd your-project-directory`

3

Start Claude Code

Run `claude` to launch

4

Complete authentication

Follow the one-time OAuth process with your Console account. You’ll need
active billing at [console.anthropic.com](https://console.anthropic.com).

---

## [​](#core-features-and-workflows) Core features and workflows

Claude Code operates directly in your terminal, understanding your project context and taking real actions. No need to manually add files to context - Claude will explore your codebase as needed. Claude Code uses `claude-3-7-sonnet-20250219` by default.

### [​](#security-and-privacy-by-design) Security and privacy by design

Your code’s security is paramount. Claude Code’s architecture ensures:

* **Direct API connection**: Your queries go straight to Anthropic’s API without intermediate servers
* **Works where you work**: Operates directly in your terminal
* **Understands context**: Maintains awareness of your entire project structure
* **Takes action**: Performs real operations like editing files and creating commits

### [​](#from-questions-to-solutions-in-seconds) From questions to solutions in seconds

```
# Ask questions about your codebase
claude
> how does our authentication system work?

# Create a commit with one command
claude commit

# Fix issues across multiple files
claude "fix the type errors in the auth module"

```

---

### [​](#initialize-your-project) Initialize your project

For first-time users, we recommend:

1. Start Claude Code with `claude`
2. Try a simple command like `summarize this project`
3. Generate a CLAUDE.md project guide with `/init`
4. Ask Claude to commit the generated CLAUDE.md file to your repository

## [​](#use-claude-code-for-common-tasks) Use Claude Code for common tasks

Claude Code operates directly in your terminal, understanding your project context and taking real actions. No need to manually add files to context - Claude will explore your codebase as needed.

### [​](#understand-unfamiliar-code) Understand unfamiliar code

```
> what does the payment processing system do?
> find where user permissions are checked
> explain how the caching layer works

```

### [​](#automate-git-operations) Automate Git operations

```
> commit my changes
> create a pr
> which commit added tests for markdown back in December?
> rebase on main and resolve any merge conflicts

```

### [​](#edit-code-intelligently) Edit code intelligently

```
> add input validation to the signup form
> refactor the logger to use the new API
> fix the race condition in the worker queue

```

### [​](#test-and-debug-your-code) Test and debug your code

```
> run tests for the auth module and fix failures
> find and fix security vulnerabilities
> explain why this test is failing

```

### [​](#encourage-deeper-thinking) Encourage deeper thinking

For complex problems, explicitly ask Claude to think more deeply:

```
> think about how we should architect the new payment service
> think hard about the edge cases in our authentication flow

```

---

## [​](#control-claude-code-with-commands) Control Claude Code with commands

### [​](#cli-commands) CLI commands

| Command | Description | Example |
| --- | --- | --- |
| `claude` | Start interactive REPL | `claude` |
| `claude "query"` | Start REPL with initial prompt | `claude "explain this project"` |
| `claude -p "query"` | Run one-off query, then exit | `claude -p "explain this function"` |
| `cat file | claude -p "query"` | Process piped content | `cat logs.txt | claude -p "explain"` |
| `claude config` | Configure settings | `claude config set --global theme dark` |
| `claude update` | Update to latest version | `claude update` |
| `claude mcp` | Configure Model Context Protocol servers | [See MCP section in tutorials](/en/docs/agents/claude-code/tutorials#set-up-model-context-protocol-mcp) |

**CLI flags**:

* `--print`: Print response without interactive mode
* `--verbose`: Enable verbose logging
* `--dangerously-skip-permissions`: Skip permission prompts (only in Docker containers without internet)

### [​](#slash-commands) Slash commands

Control Claude’s behavior within a session:

| Command | Purpose |
| --- | --- |
| `/bug` | Report bugs (sends conversation to Anthropic) |
| `/clear` | Clear conversation history |
| `/compact` | Compact conversation to save context space |
| `/config` | View/modify configuration |
| `/cost` | Show token usage statistics |
| `/doctor` | Checks the health of your Claude Code installation |
| `/help` | Get usage help |
| `/init` | Initialize project with CLAUDE.md guide |
| `/login` | Switch Anthropic accounts |
| `/logout` | Sign out from your Anthropic account |
| `/pr_comments` | View pull request comments |
| `/review` | Request code review |
| `/terminal-setup` | Install Shift+Enter key binding for newlines (iTerm2 and VSCode only) |

## [​](#manage-permissions-and-security) Manage permissions and security

Claude Code uses a tiered permission system to balance power and safety:

| Tool Type | Example | Approval Required | ”Yes, don’t ask again” Behavior |
| --- | --- | --- | --- |
| Read-only | File reads, LS, Grep | No | N/A |
| Bash Commands | Shell execution | Yes | Permanently per project directory and command |
| File Modification | Edit/write files | Yes | Until session end |

### [​](#tools-available-to-claude) Tools available to Claude

Claude Code has access to a set of powerful tools that help it understand and modify your codebase:

| Tool | Description | Permission Required |
| --- | --- | --- |
| **AgentTool** | Runs a sub-agent to handle complex, multi-step tasks | No |
| **BashTool** | Executes shell commands in your environment | Yes |
| **GlobTool** | Finds files based on pattern matching | No |
| **GrepTool** | Searches for patterns in file contents | No |
| **LSTool** | Lists files and directories | No |
| **FileReadTool** | Reads the contents of files | No |
| **FileEditTool** | Makes targeted edits to specific files | Yes |
| **FileWriteTool** | Creates or overwrites files | Yes |
| **NotebookReadTool** | Reads and displays Jupyter notebook contents | No |
| **NotebookEditTool** | Modifies Jupyter notebook cells | Yes |

### [​](#protect-against-prompt-injection) Protect against prompt injection

Prompt injection is a technique where an attacker attempts to override or manipulate an AI assistant’s instructions by inserting malicious text. Claude Code includes several safeguards against these attacks:

* **Permission system**: Sensitive operations require explicit approval
* **Context-aware analysis**: Detects potentially harmful instructions by analyzing the full request
* **Input sanitization**: Prevents command injection by processing user inputs
* **Command blocklist**: Blocks risky commands that fetch arbitrary content from the web like `curl` and `wget`

**Best practices for working with untrusted content**:

1. Review suggested commands before approval
2. Avoid piping untrusted content directly to Claude
3. Verify proposed changes to critical files
4. Report suspicious behavior with `/bug`

While these protections significantly reduce risk, no system is completely
immune to all attacks. Always maintain good security practices when working
with any AI tool.

### [​](#configure-network-access) Configure network access

Claude Code requires access to:

* api.anthropic.com
* statsig.anthropic.com
* sentry.io

Allowlist these URLs when using Claude Code in containerized environments.

---

## [​](#configure-claude-code) Configure Claude Code

Configure Claude Code by running `claude config` in your terminal, or the `/config` command when using the interactive REPL.

### [​](#configuration-options) Configuration options

Claude Code supports global and project-level configuration.

To manage your configurations, use the following commands:

* List settings: `claude config list`
* See a setting: `claude config get <key>`
* Change a setting: `claude config set <key> <value>`
* Push to a setting (for lists): `claude config add <key> <value>`
* Remove from a setting (for lists): `claude config remove <key> <value>`

By default `config` changes your project configuration. To manage your global configuration, use the `--global` (or `-g`) flag.

#### [​](#global-configuration) Global configuration

To set a global configuration, use `claude config set -g <key> <value>`:

| Key | Value | Description |
| --- | --- | --- |
| `autoUpdaterStatus` | `disabled` or `enabled` | Enable or disable the auto-updater (default: `enabled`) |
| `preferredNotifChannel` | `iterm2`, `iterm2_with_bell`, `terminal_bell`, or `notifications_disabled` | Where you want to receive notifications (default: `iterm2`) |
| `theme` | `dark`, `light`, `light-daltonized`, or `dark-daltonized` | Color theme |
| `verbose` | `true` or `false` | Whether to show full bash and command outputs (default: `false`) |

### [​](#auto-updater-permission-options) Auto-updater permission options

When Claude Code detects that it doesn’t have sufficient permissions to write to your global npm prefix directory (required for automatic updates), you’ll see a warning that points to this documentation page. For detailed solutions to auto-updater issues, see the [troubleshooting guide](/en/docs/agents-and-tools/claude-code/troubleshooting#auto-updater-issues).

#### [​](#recommended-create-a-new-user-writable-npm-prefix) Recommended: Create a new user-writable npm prefix

```
# First, save a list of your existing global packages for later migration
npm list -g --depth=0 > ~/npm-global-packages.txt

# Create a directory for your global packages
mkdir -p ~/.npm-global

# Configure npm to use the new directory path
npm config set prefix ~/.npm-global

# Note: Replace ~/.bashrc with ~/.zshrc, ~/.profile, or other appropriate file for your shell
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc

# Apply the new PATH setting
source ~/.bashrc

# Now reinstall Claude Code in the new location
npm install -g @anthropic-ai/claude-code

# Optional: Reinstall your previous global packages in the new location
# Look at ~/npm-global-packages.txt and install packages you want to keep
# npm install -g package1 package2 package3...

```

**Why we recommend this option:**

* Avoids modifying system directory permissions
* Creates a clean, dedicated location for your global npm packages
* Follows security best practices

Since Claude Code is actively developing, we recommend setting up auto-updates using the recommended option above.

#### [​](#disabling-the-auto-updater) Disabling the auto-updater

If you prefer to disable the auto-updater instead of fixing permissions, you can use:

```
claude config set -g autoUpdaterStatus disabled

```

#### [​](#project-configuration) Project configuration

Manage project configuration with `claude config set <key> <value>` (without the `-g` flag):

| Key | Value | Description |
| --- | --- | --- |
| `allowedTools` | array of tools | Which tools can run without manual approval |
| `ignorePatterns` | array of glob strings | Which files/directories are ignored when using tools |

For example:

```
# Let npm test to run without approval
claude config add allowedTools "Bash(npm test)"

# Let npm test and any of its sub-commands to run without approval
claude config add allowedTools "Bash(npm test:*)"

# Instruct Claude to ignore node_modules
claude config add ignorePatterns node_modules
claude config add ignorePatterns "node_modules/**"

```

### [​](#optimize-your-terminal-setup) Optimize your terminal setup

Claude Code works best when your terminal is properly configured. Follow these guidelines to optimize your experience.

**Supported shells**:

* Bash
* Zsh
* Fish

#### [​](#themes-and-appearance) Themes and appearance

Claude cannot control the theme of your terminal. That’s handled by your terminal application. You can match Claude Code’s theme to your terminal during onboarding or any time via the `/config` command

#### [​](#line-breaks) Line breaks

You have several options for entering linebreaks into Claude Code:

* **Quick escape**: Type `\` followed by Enter to create a newline
* **Keyboard shortcut**: Press Option+Enter (Meta+Enter) with proper configuration

To set up Option+Enter in your terminal:

**For Mac Terminal.app:**

1. Open Settings → Profiles → Keyboard
2. Check “Use Option as Meta Key”

**For iTerm2 and VSCode terminal:**

1. Open Settings → Profiles → Keys
2. Under General, set Left/Right Option key to “Esc+”

**Tip for iTerm2 and VSCode users**: Run `/terminal-setup` within Claude Code to automatically configure Shift+Enter as a more intuitive alternative.

#### [​](#notification-setup) Notification setup

Never miss when Claude completes a task with proper notification configuration:

##### Terminal bell notifications

Enable sound alerts when tasks complete:

```
claude config set --global preferredNotifChannel terminal_bell

```

**For macOS users**: Don’t forget to enable notification permissions in System Settings → Notifications → [Your Terminal App].

##### iTerm 2 system notifications

For iTerm 2 alerts when tasks complete:

1. Open iTerm 2 Preferences
2. Navigate to Profiles → Terminal
3. Enable “Silence bell” and “Send notification when idle”
4. Set your preferred notification delay

Note that these notifications are specific to iTerm 2 and not available in the default macOS Terminal.

#### [​](#handling-large-inputs) Handling large inputs

When working with extensive code or long instructions:

* **Avoid direct pasting**: Claude Code may struggle with very long pasted content
* **Use file-based workflows**: Write content to a file and ask Claude to read it
* **Be aware of VS Code limitations**: The VS Code terminal is particularly prone to truncating long pastes

By configuring these settings, you’ll create a smoother, more efficient workflow with Claude Code.

---

## [​](#manage-costs-effectively) Manage costs effectively

Claude Code consumes tokens for each interaction. Typical usage costs range from $5-10 per developer per day, but can exceed $100 per hour during intensive use.

### [​](#track-your-costs) Track your costs

* Use `/cost` to see current session usage
* Review cost summary displayed when exiting
* Check historical usage in [Anthropic Console](https://console.anthropic.com)
* Set [Spend limits](https://console.anthropic.com/settings/limits)

### [​](#reduce-token-usage) Reduce token usage

* **Compact conversations:** Use `/compact` when context gets large
* **Write specific queries:** Avoid vague requests that trigger unnecessary scanning
* **Break down complex tasks:** Split large tasks into focused interactions
* **Clear history between tasks:** Use `/clear` to reset context

Costs can vary significantly based on:

* Size of codebase being analyzed
* Complexity of queries
* Number of files being searched or modified
* Length of conversation history
* Frequency of compacting conversations

For team deployments, we recommend starting with a small pilot group to
establish usage patterns before wider rollout.

---

## [​](#use-with-third-party-apis) Use with third-party APIs

Claude Code requires access to both Claude 3.7 Sonnet and Claude 3.5 Haiku
models, regardless of which API provider you use.

### [​](#connect-to-amazon-bedrock) Connect to Amazon Bedrock

```
CLAUDE_CODE_USE_BEDROCK=1

```

Optional: Override the default model (Claude 3.7 Sonnet is used by default):

```
ANTHROPIC_MODEL='us.anthropic.claude-3-7-sonnet-20250219-v1:0'

```

If you don’t have prompt caching enabled, also set:

```
DISABLE_PROMPT_CACHING=1

```

Requires standard AWS SDK credentials (e.g., `~/.aws/credentials` or relevant environment variables like `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Contact Amazon Bedrock for prompt caching for reduced costs and higher rate limits.

Users will need access to both Claude 3.7 Sonnet and Claude 3.5 Haiku models
in their AWS account. If you have a model access role, you may need to request
access to these models if they’re not already available.

### [​](#connect-to-google-vertex-ai) Connect to Google Vertex AI

```
CLAUDE_CODE_USE_VERTEX=1
CLOUD_ML_REGION=us-east5
ANTHROPIC_VERTEX_PROJECT_ID=your-project-id

```

Claude Code on Vertex AI currently only supports the `us-east5` region. Make
sure your project has quota allocated in this specific region.

Users will need access to both Claude 3.7 Sonnet and Claude 3.5 Haiku models
in their Vertex AI project.

Requires standard GCP credentials configured through google-auth-library. For the best experience, contact Google for heightened rate limits.

---

## [​](#development-container-reference-implementation) Development container reference implementation

Claude Code provides a development container configuration for teams that need consistent, secure environments. This preconfigured [devcontainer setup](https://code.visualstudio.com/docs/devcontainers/containers) works seamlessly with VS Code’s Remote - Containers extension and similar tools.

The container’s enhanced security measures (isolation and firewall rules) allow you to run `claude --dangerously-skip-permissions` to bypass permission prompts for unattended operation. We’ve included a [reference implementation](https://github.com/anthropics/claude-code/tree/main/.devcontainer) that you can customize for your needs.

While the devcontainer provides substantial protections, no system is
completely immune to all attacks. Always maintain good security practices and
monitor Claude’s activities.

### [​](#key-features) Key features

* **Production-ready Node.js**: Built on Node.js 20 with essential development dependencies
* **Security by design**: Custom firewall restricting network access to only necessary services
* **Developer-friendly tools**: Includes git, ZSH with productivity enhancements, fzf, and more
* **Seamless VS Code integration**: Pre-configured extensions and optimized settings
* **Session persistence**: Preserves command history and configurations between container restarts
* **Works everywhere**: Compatible with macOS, Windows, and Linux development environments

### [​](#getting-started-in-4-steps) Getting started in 4 steps

1. Install VS Code and the Remote - Containers extension
2. Clone the [Claude Code reference implementation](https://github.com/anthropics/claude-code/tree/main/.devcontainer) repository
3. Open the repository in VS Code
4. When prompted, click “Reopen in Container” (or use Command Palette: Cmd+Shift+P → “Remote-Containers: Reopen in Container”)

### [​](#configuration-breakdown) Configuration breakdown

The devcontainer setup consists of three primary components:

* [**devcontainer.json**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/devcontainer.json): Controls container settings, extensions, and volume mounts
* [**Dockerfile**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/Dockerfile): Defines the container image and installed tools
* [**init-firewall.sh**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh): Establishes network security rules

### [​](#security-features) Security features

The container implements a multi-layered security approach with its firewall configuration:

* **Precise access control**: Restricts outbound connections to whitelisted domains only (npm registry, GitHub, Anthropic API, etc.)
* **Default-deny policy**: Blocks all other external network access
* **Startup verification**: Validates firewall rules when the container initializes
* **Isolation**: Creates a secure development environment separated from your main system

### [​](#customization-options) Customization options

The devcontainer configuration is designed to be adaptable to your needs:

* Add or remove VS Code extensions based on your workflow
* Modify resource allocations for different hardware environments
* Adjust network access permissions
* Customize shell configurations and developer tooling

---

## [​](#next-steps) Next steps

[## Claude Code tutorials

Step-by-step guides for common tasks](/en/docs/agents-and-tools/claude-code/tutorials)[## Troubleshooting

Solutions for common issues with Claude Code](/en/docs/agents-and-tools/claude-code/troubleshooting)[## Reference implementation

Clone our development container reference implementation.](https://github.com/anthropics/claude-code/tree/main/.devcontainer)

---

## [​](#license-and-data-usage) License and data usage

Claude Code is provided as a Beta research preview under Anthropic’s [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms).

### [​](#how-we-use-your-data) How we use your data

We aim to be fully transparent about how we use your data. We may use feedback to improve our products and services, but we will not train generative models using your feedback from Claude Code. Given their potentially sensitive nature, we store user feedback transcripts for only 30 days.

#### [​](#feedback-transcripts) Feedback transcripts

If you choose to send us feedback about Claude Code, such as transcripts of your usage, Anthropic may use that feedback to debug related issues and improve Claude Code’s functionality (e.g., to reduce the risk of similar bugs occurring in the future). We will not train generative models using this feedback.

### [​](#privacy-safeguards) Privacy safeguards

We have implemented several safeguards to protect your data, including limited retention periods for sensitive information, restricted access to user session data, and clear policies against using feedback for model training.

For full details, please review our [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms) and [Privacy Policy](https://www.anthropic.com/legal/privacy).

### [​](#license) License

© Anthropic PBC. All rights reserved. Use is subject to Anthropic’s [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms).

Was this page helpful?

YesNo

[Embeddings](/en/docs/build-with-claude/embeddings)[Claude Code tutorials](/en/docs/agents-and-tools/claude-code/tutorials)

[x](https://x.com/AnthropicAI)[linkedin](https://www.linkedin.com/company/anthropicresearch)

On this page

* [Before you begin](#before-you-begin)
* [Check system requirements](#check-system-requirements)
* [Install and authenticate](#install-and-authenticate)
* [Core features and workflows](#core-features-and-workflows)
* [Security and privacy by design](#security-and-privacy-by-design)
* [From questions to solutions in seconds](#from-questions-to-solutions-in-seconds)
* [Initialize your project](#initialize-your-project)
* [Use Claude Code for common tasks](#use-claude-code-for-common-tasks)
* [Understand unfamiliar code](#understand-unfamiliar-code)
* [Automate Git operations](#automate-git-operations)
* [Edit code intelligently](#edit-code-intelligently)
* [Test and debug your code](#test-and-debug-your-code)
* [Encourage deeper thinking](#encourage-deeper-thinking)
* [Control Claude Code with commands](#control-claude-code-with-commands)
* [CLI commands](#cli-commands)
* [Slash commands](#slash-commands)
* [Manage permissions and security](#manage-permissions-and-security)
* [Tools available to Claude](#tools-available-to-claude)
* [Protect against prompt injection](#protect-against-prompt-injection)
* [Configure network access](#configure-network-access)
* [Configure Claude Code](#configure-claude-code)
* [Configuration options](#configuration-options)
* [Global configuration](#global-configuration)
* [Auto-updater permission options](#auto-updater-permission-options)
* [Recommended: Create a new user-writable npm prefix](#recommended-create-a-new-user-writable-npm-prefix)
* [Disabling the auto-updater](#disabling-the-auto-updater)
* [Project configuration](#project-configuration)
* [Optimize your terminal setup](#optimize-your-terminal-setup)
* [Themes and appearance](#themes-and-appearance)
* [Line breaks](#line-breaks)
* [Notification setup](#notification-setup)
* [Handling large inputs](#handling-large-inputs)
* [Manage costs effectively](#manage-costs-effectively)
* [Track your costs](#track-your-costs)
* [Reduce token usage](#reduce-token-usage)
* [Use with third-party APIs](#use-with-third-party-apis)
* [Connect to Amazon Bedrock](#connect-to-amazon-bedrock)
* [Connect to Google Vertex AI](#connect-to-google-vertex-ai)
* [Development container reference implementation](#development-container-reference-implementation)
* [Key features](#key-features)
* [Getting started in 4 steps](#getting-started-in-4-steps)
* [Configuration breakdown](#configuration-breakdown)
* [Security features](#security-features)
* [Customization options](#customization-options)
* [Next steps](#next-steps)
* [License and data usage](#license-and-data-usage)
* [How we use your data](#how-we-use-your-data)
* [Feedback transcripts](#feedback-transcripts)
* [Privacy safeguards](#privacy-safeguards)
* [License](#license)