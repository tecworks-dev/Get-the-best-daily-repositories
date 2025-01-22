# Migrating from Git to Evo

This guide will help you transition from Git to Evo, explaining key differences and providing equivalent commands.

## Conceptual Differences

### Branches vs Workspaces

Git uses branches as long-lived lines of development. Evo uses workspaces, which are:
- Ephemeral by design
- Focused on single features/tasks
- Meant to be merged and cleaned up quickly

### Staging Area

Both Git and Evo have staging areas, but Evo's is:
- More intuitive with structural awareness
- Supports partial file staging by default
- Provides better feedback about staged changes

### Merging

Evo's merge system is:
- More intelligent with structural merges
- Better at handling renames
- Designed to reduce conflicts
- More interactive when conflicts do occur

## Command Mapping

### Basic Commands

| Git Command | Evo Command | Notes |
|------------|-------------|--------|
| `git init` | `evo init` | Similar functionality |
| `git status` | `evo status` | Clearer output format |
| `git add` | `evo stage` | More intuitive name |
| `git commit` | `evo commit` | Similar functionality |
| `git log` | `evo log` | Enhanced verification info |
| `git push` | `evo sync` | Combines push/pull |
| `git pull` | `evo sync` | Combines push/pull |

### Branch Operations

| Git Command | Evo Command | Notes |
|------------|-------------|--------|
| `git branch` | `evo workspace list` | List workspaces |
| `git checkout -b` | `evo workspace create` | Create workspace |
| `git checkout` | `evo workspace switch` | Switch workspace |
| `git branch -d` | `evo workspace delete` | Delete workspace |
| `git merge` | `evo workspace merge` | Merge workspace |

### Remote Operations

| Git Command | Evo Command | Notes |
|------------|-------------|--------|
| `git remote add` | `evo remote add` | Similar functionality |
| `git remote -v` | `evo remote list` | List remotes |
| `git fetch` | `evo sync --no-push` | Get remote changes |
| `git push -u origin` | `evo sync` | Push to remote |

## Migration Steps

### 1. Prepare Your Repository

```bash
# In your Git repository
git commit -am "Final Git commit"
git push
```

### 2. Initialize Evo

```bash
# Create new Evo repository
evo init
evo remote add origin <your-repo-url>
```

### 3. Import History (Optional)

```bash
# Evo provides a Git import tool
evo import git .
```

### 4. Update Workflows

1. Replace branch-based workflows with workspace-based ones
2. Update CI/CD pipelines to use Evo commands
3. Update team documentation

## Best Practices When Migrating

1. **Start Fresh**
   - Consider starting with a clean Evo repository for new features
   - Keep the Git repository as archive if needed

2. **Team Training**
   - Ensure team understands workspace concept
   - Practice common workflows together
   - Review new merge strategies

3. **Gradual Migration**
   - Migrate one team/project at a time
   - Run both systems in parallel initially
   - Set a cut-off date for Git usage

## Common Pitfalls

1. **Trying to Use Git Workflows**
   - Embrace Evo's workspace model
   - Don't try to recreate long-lived branches

2. **Ignoring Structural Merges**
   - Take advantage of Evo's smart merge capabilities
   - Use appropriate file formats for configs

3. **Skipping Verification**
   - Enable commit signing from the start
   - Use Evo's built-in verification features

## Tips for Teams

1. **Establish Conventions**
   - Workspace naming patterns
   - Commit message formats
   - Merge strategies

2. **Update Tools**
   - IDE integrations
   - CI/CD pipelines
   - Code review processes

3. **Document Everything**
   - Keep track of migration decisions
   - Document new workflows
   - Create team-specific guides

## Need Help?

- Check our [Command Reference](../commands.md)
- Visit our [Getting Started Guide](../getting-started.md)
- Open an issue on our GitHub repository
- Join our community chat
