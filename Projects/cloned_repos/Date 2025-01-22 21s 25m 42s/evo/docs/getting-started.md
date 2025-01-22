# Getting Started with Evo

Welcome to Evo! This guide will help you get up and running with Evo's modern approach to version control.

## Installation

### Prerequisites
- Go 1.20 or higher
- A terminal emulator
- (Optional) Git for migration purposes

### Installing Evo

```bash
go install github.com/crazywolf132/evo/cmd/evo@latest
```

Verify your installation:
```bash
evo --version
```

## Basic Workflow

### 1. Creating a New Repository

Initialize a new Evo repository in your project directory:
```bash
cd your-project
evo init
```

This creates a `.evo` directory to store version control information.

### 2. Working with Workspaces

Instead of branches, Evo uses workspaces for feature development:

```bash
# Create a new workspace
evo workspace create feature-name

# List all workspaces
evo workspace list

# Switch to a workspace
evo workspace switch feature-name

# Delete a workspace (after merging)
evo workspace delete feature-name
```

### 3. Making Changes

```bash
# Check status of your changes
evo status

# Stage specific files
evo stage file1.txt file2.txt

# Stage all changes
evo stage .

# Commit your changes
evo commit -m "Add new feature"

# View commit history
evo log
```

### 4. Merging Changes

```bash
# Merge your workspace back to main
evo workspace merge

# Handle any conflicts if they arise
evo merge --continue  # after resolving conflicts
```

### 5. Syncing with Remote

```bash
# Add a remote server
evo remote add origin https://your-server.com/repo

# Push your changes
evo sync
```

## Working with Teams

### Setting Up Authentication

```bash
# Configure your identity
evo config set user.name "Your Name"
evo config set user.email "your.email@example.com"

# Set up commit signing (optional)
evo keys generate
```

### Collaborating

```bash
# Get latest changes
evo sync

# Create a workspace for collaboration
evo workspace create shared-feature

# Push workspace for others to see
evo workspace push shared-feature
```

## Advanced Features

### Large File Support

Large files are automatically handled:
```bash
# Configure large file threshold
evo config set core.largeFileThreshold "5MB"
```

### Structural Merges

Evo automatically handles structural merges for JSON and YAML files. No additional configuration needed!

### Commit Signing

```bash
# Enable commit signing
evo config set commit.sign true

# Sign all commits in a workspace
evo workspace sign feature-name
```

## Best Practices

1. **Keep Workspaces Short-lived**
   - Create them for specific features
   - Merge them as soon as the feature is complete

2. **Commit Often**
   - Make small, focused commits
   - Use clear commit messages

3. **Sync Regularly**
   - Pull changes from remote frequently
   - Push your changes when they're ready for others

4. **Use Meaningful Names**
   - Give workspaces descriptive names
   - Use clear commit messages

## Troubleshooting

### Common Issues

1. **Merge Conflicts**
   ```bash
   # Abort current merge
   evo merge --abort
   
   # Start fresh
   evo workspace create feature-name-new
   ```

2. **Sync Issues**
   ```bash
   # Force sync (use with caution)
   evo sync --force
   ```

3. **Workspace Problems**
   ```bash
   # Reset workspace to clean state
   evo workspace reset
   ```

## Next Steps

- Read the [Command Reference](./commands.md) for detailed information about all available commands
- Check out our [migration guides](./migrations/) if you're moving from another VCS
- Visit our [GitHub repository](https://github.com/crazywolf132/evo) for the latest updates

Need help? Open an issue on our GitHub repository!
