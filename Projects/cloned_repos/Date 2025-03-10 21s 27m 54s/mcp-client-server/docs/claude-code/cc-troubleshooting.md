Claude Code troubleshooting - Anthropic

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

Claude Code troubleshooting

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

# Claude Code troubleshooting

Solutions for common issues with Claude Code installation and usage.

## [​](#common-installation-issues) Common installation issues

### [​](#linux-permission-issues) Linux permission issues

When installing Claude Code with npm, you may encounter permission errors if your npm global prefix is not user writable (eg. `/usr`, or `/use/local`).

#### [​](#recommended-solution-create-a-user-writable-npm-prefix) Recommended solution: Create a user-writable npm prefix

The safest approach is to configure npm to use a directory within your home folder:

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

```

This solution is recommended because it:

* Avoids modifying system directory permissions
* Creates a clean, dedicated location for your global npm packages
* Follows security best practices

#### [​](#system-recovery-if-you-have-run-commands-that-change-ownership-and-permissions-of-system-files-or-similar) System Recovery: If you have run commands that change ownership and permissions of system files or similar

If you’ve already run a command that changed system directory permissions (such as `sudo chown -R $USER:$(id -gn) /usr && sudo chmod -R u+w /usr`) and your system is now broken (for example, if you see `sudo: /usr/bin/sudo must be owned by uid 0 and have the setuid bit set`), you’ll need to perform recovery steps.

##### Ubuntu/Debian Recovery Method:

1. While rebooting, hold **SHIFT** to access the GRUB menu
2. Select “Advanced options for Ubuntu/Debian”
3. Choose the recovery mode option
4. Select “Drop to root shell prompt”
5. Remount the filesystem as writable:

   ```
   mount -o remount,rw /

   ```
6. Fix permissions:

   ```
   # Restore root ownership
   chown -R root:root /usr
   chmod -R 755 /usr

   # Ensure /usr/local is owned by your user for npm packages
   chown -R YOUR_USERNAME:YOUR_USERNAME /usr/local

   # Set setuid bit for critical binaries
   chmod u+s /usr/bin/sudo
   chmod 4755 /usr/bin/sudo
   chmod u+s /usr/bin/su
   chmod u+s /usr/bin/passwd
   chmod u+s /usr/bin/newgrp
   chmod u+s /usr/bin/gpasswd
   chmod u+s /usr/bin/chsh
   chmod u+s /usr/bin/chfn

   # Fix sudo configuration
   chown root:root /usr/libexec/sudo/sudoers.so
   chmod 4755 /usr/libexec/sudo/sudoers.so
   chown root:root /etc/sudo.conf
   chmod 644 /etc/sudo.conf

   ```
7. Reinstall affected packages (optional but recommended):

   ```
   # Save list of installed packages
   dpkg --get-selections > /tmp/installed_packages.txt

   # Reinstall them
   awk '{print $1}' /tmp/installed_packages.txt | xargs -r apt-get install --reinstall -y

   ```
8. Reboot:

   ```
   reboot

   ```

##### Alternative Live USB Recovery Method:

If the recovery mode doesn’t work, you can use a live USB:

1. Boot from a live USB (Ubuntu, Debian, or any Linux distribution)
2. Find your system partition:

   ```
   lsblk

   ```
3. Mount your system partition:

   ```
   sudo mount /dev/sdXY /mnt  # replace sdXY with your actual system partition

   ```
4. If you have a separate boot partition, mount it too:

   ```
   sudo mount /dev/sdXZ /mnt/boot  # if needed

   ```
5. Chroot into your system:

   ```
   # For Ubuntu/Debian:
   sudo chroot /mnt

   # For Arch-based systems:
   sudo arch-chroot /mnt

   ```
6. Follow steps 6-8 from the Ubuntu/Debian recovery method above

After restoring your system, follow the recommended solution above to set up a user-writable npm prefix.

## [​](#auto-updater-issues) Auto-updater issues

If Claude Code can’t update automatically, it may be due to permission issues with your npm global prefix directory. Follow the [recommended solution](/_sites/docs.anthropic.com/en/docs/agents-and-tools/claude-code/troubleshooting#recommended-solution-create-a-user-writable-npm-prefix) above to fix this.

If you prefer to disable the auto-updater instead, you can use:

```
claude config set -g autoUpdaterStatus disabled

```

## [​](#permissions-and-authentication) Permissions and authentication

### [​](#repeated-permission-prompts) Repeated permission prompts

If you find yourself repeatedly approving the same commands, you can allow specific tools to run without approval:

```
# Let npm test run without approval
claude config add allowedTools "Bash(npm test)"

# Let npm test and any of its sub-commands run without approval
claude config add allowedTools "Bash(npm test:*)"

```

### [​](#authentication-issues) Authentication issues

If you’re experiencing authentication problems:

1. Run `/logout` to sign out completely
2. Close Claude Code
3. Restart with `claude` and complete the authentication process again

If problems persist, try:

```
rm -rf ~/.config/claude-code/auth.json
claude

```

This removes your stored authentication information and forces a clean login.

## [​](#performance-and-stability) Performance and stability

### [​](#high-cpu-or-memory-usage) High CPU or memory usage

Claude Code is designed to work with most development environments, but may consume significant resources when processing large codebases. If you’re experiencing performance issues:

1. Use `/compact` regularly to reduce context size
2. Close and restart Claude Code between major tasks
3. Consider adding large build directories to your `.gitignore` and `.claudeignore` files

### [​](#command-hangs-or-freezes) Command hangs or freezes

If Claude Code seems unresponsive:

1. Press Ctrl+C to attempt to cancel the current operation
2. If unresponsive, you may need to close the terminal and restart
3. For persistent issues, run Claude with verbose logging: `claude --verbose`

## [​](#getting-more-help) Getting more help

If you’re experiencing issues not covered here:

1. Use the `/bug` command within Claude Code to report problems directly to Anthropic
2. Check the [GitHub repository](https://github.com/anthropics/claude-code) for known issues
3. Run `/doctor` to check the health of your Claude Code installation

Was this page helpful?

YesNo

[Claude Code tutorials](/en/docs/agents-and-tools/claude-code/tutorials)[Computer use (beta)](/en/docs/agents-and-tools/computer-use)

[x](https://x.com/AnthropicAI)[linkedin](https://www.linkedin.com/company/anthropicresearch)

On this page

* [Common installation issues](#common-installation-issues)
* [Linux permission issues](#linux-permission-issues)
* [Recommended solution: Create a user-writable npm prefix](#recommended-solution-create-a-user-writable-npm-prefix)
* [System Recovery: If you have run commands that change ownership and permissions of system files or similar](#system-recovery-if-you-have-run-commands-that-change-ownership-and-permissions-of-system-files-or-similar)
* [Auto-updater issues](#auto-updater-issues)
* [Permissions and authentication](#permissions-and-authentication)
* [Repeated permission prompts](#repeated-permission-prompts)
* [Authentication issues](#authentication-issues)
* [Performance and stability](#performance-and-stability)
* [High CPU or memory usage](#high-cpu-or-memory-usage)
* [Command hangs or freezes](#command-hangs-or-freezes)
* [Getting more help](#getting-more-help)