# Evo Command Reference

This document provides a comprehensive reference for all Evo commands, their options, and usage examples.

## Core Commands

### init
Initialize a new Evo repository.

```bash
evo init [options]
```

Options:
- `--bare`: Create a bare repository
- `--template=<path>`: Specify a custom template directory

### status
Show the working tree status.

```bash
evo status [options]
```

Options:
- `--short`: Give the output in short format
- `--json`: Output status in JSON format

### stage
Add file contents to the staging area.

```bash
evo stage [options] [--] [<pathspec>...]
```

Options:
- `-A, --all`: Stage all changes
- `-p, --patch`: Interactively choose hunks to stage
- `--dry-run`: Show what would be staged

### commit
Record changes to the repository.

```bash
evo commit [options] [--] [<pathspec>...]
```

Options:
- `-m, --message <message>`: Use the given message as the commit message
- `-s, --sign`: Sign the commit
- `--amend`: Amend the previous commit
- `--no-verify`: Skip pre-commit hooks

### log
Show commit logs.

```bash
evo log [options] [<revision-range>]
```

Options:
- `--oneline`: Show one line per commit
- `--graph`: Draw a text-based graphical representation
- `--verify`: Show signature verification status
- `-n <number>`: Limit output to <number> commits

## Workspace Commands

### workspace create
Create a new workspace.

```bash
evo workspace create <name> [options]
```

Options:
- `--from <commit>`: Create from specific commit
- `--empty`: Create an empty workspace

### workspace switch
Switch to a different workspace.

```bash
evo workspace switch <name>
```

### workspace merge
Merge workspace changes back to main.

```bash
evo workspace merge [options]
```

Options:
- `--no-ff`: Create a merge commit even when fast-forward is possible
- `--squash`: Squash all commits into one
- `--verify`: Verify all commits before merging

### workspace list
List all workspaces.

```bash
evo workspace list [options]
```

Options:
- `--verbose`: Show additional details
- `--json`: Output in JSON format

## Remote Operations

### remote
Manage remote repositories.

```bash
evo remote <subcommand>
```

Subcommands:
- `add <name> <url>`: Add a remote repository
- `remove <name>`: Remove a remote repository
- `list`: List remote repositories

### sync
Synchronize with remote repository.

```bash
evo sync [options]
```

Options:
- `--force`: Force sync even with conflicts
- `--dry-run`: Show what would be synced
- `--verify`: Verify signatures during sync

## Configuration

### config
Get and set repository or global options.

```bash
evo config [options] <name> [<value>]
```

Options:
- `--global`: Use global config file
- `--local`: Use repository config file
- `--list`: List all variables

### keys
Manage signing keys.

```bash
evo keys <subcommand>
```

Subcommands:
- `generate`: Generate new signing key
- `list`: List all keys
- `export`: Export public key
- `import`: Import key

## Advanced Commands

### merge
Join two or more development histories together.

```bash
evo merge [options] <workspace>
```

Options:
- `--abort`: Abort the current merge
- `--continue`: Continue the current merge
- `--strategy <strategy>`: Use specific merge strategy

### revert
Revert changes by commit.

```bash
evo revert [options] <commit>...
```

Options:
- `--no-commit`: Don't create a commit
- `--edit`: Edit the revert message

### server
Start an Evo server for remote operations.

```bash
evo server [options]
```

Options:
- `--port <port>`: Specify port number
- `--host <host>`: Specify host address
- `--auth`: Enable authentication
- `--cert <path>`: Path to SSL certificate

## Environment Variables

- `EVO_DIR`: Override the .evo directory location
- `EVO_EDITOR`: Specify editor for commit messages
- `EVO_CONFIG`: Specify alternative config file
- `EVO_SERVER`: Default server URL
- `EVO_KEY_PATH`: Path to signing keys

## Exit Codes

- 0: Success
- 1: Generic error
- 2: Invalid usage
- 3: Merge conflict
- 4: Network error
- 5: Permission denied

## Configuration Files

### Global Config
Located at `~/.config/evo/config.json`

### Repository Config
Located at `.evo/config/config.json`

## See Also

- [Getting Started Guide](./getting-started.md)
- [Migration Guides](./migrations/)
- [Design Document](../DESIGN.md)
