# Evo Design Document

Below is a high-level overview of Evo's goals and design choices, intended to help new contributors understand the project and see how everything fits together.

Evo is an offline-first, tree-based version control system (VCS) that aims to improve on traditional workflows by providing:

1. User-friendly commands (no steep learning curve)
2. Ephemeral workspaces instead of complicated branch pointers
3. Advanced merges, including structural merges for JSON/YAML
4. A built-in HTTP server for push/pull
5. Optional commit signing and verification with Ed25519
6. Large file support (LFS-like mechanism)
7. Fine-grained concurrency with subdirectory/file-level locks
8. Focus on clarity and simplicity over opaque complexities

The system is implemented in Go, with an easily maintainable, extensible architecture.

## Key Goals

### 1. Simplicity & Approachability
- Minimize mental overhead: commands like `init`, `commit`, `sync`, `revert`, `workspace merge` should cover 90%+ of day-to-day tasks
- Provide straightforward conflict resolution steps and structural merges for common file types

### 2. Offline-First Distribution
- Every developer has a complete local repository, including all commits and file data
- The network is only required for syncing changes to a remote server, so one can commit, revert, and merge entirely offline

### 3. Production-Grade Reliability
- DAG-based commits with cryptographic hashes ensure integrity
- Fine-grained locking for concurrency support
- Real revert logic that inverts actual file diffs
- Built-in server for push/pull, user auth, and hosting

### 4. Enterprise & Open-Source Friendly
- Secure commit signing & verification
- Large file support
- Extensible design: easy to add new merge strategies, custom authentication backends, or specialized deployment configurations

## Core Design Choices

### 1. Custom .evo/ Format

Unlike Git's .git/, Evo maintains its own repository structure:

- **objects/**
  - Stores commits (`<commit-hash>.json`), tree objects (`<tree-hash>.json`), and binary blobs (`blobs/<blob-hash>`)
- **refs/**
  - Contains reference files (like `refs/origin`, `refs/main`) if you choose to keep named references
- **workspaces/**
  - Holds ephemeral workspace references (`workspaces/<workspace-name>`)
- **staging/**
  - Tracks partial staging information if a user commits with `--partial`
- **keys/**
  - Stores encryption-protected private keys and public keys for signing
- **config/**
  - Houses user settings (like `user.json` for name/email), remote URLs, or other repository-level config

### 2. Tree-Based Commits

Each commit references a tree object, which is effectively a snapshot of every file path mapped to a blob hash. This design:
- Makes it straightforward to see which files changed between commits by comparing trees
- Allows partial staging, revert operations, and merges at a file or even sub-file level if we store structural diffs

### 3. Renaming "HEAD"

To differentiate from Git's "HEAD," Evo uses `ACTIVE` as the name of the current commit reference. This is purely cosmetic but emphasizes that we're building something distinct from Git.

### 4. Ephemeral Workspaces

Instead of long-lived branches with numerous merges, Evo uses ephemeral "workspaces":
- Users create a workspace (`evo workspace create feature-xyz`) which references the current `ACTIVE` commit
- They switch to the workspace to do local commits
- They eventually merge the workspace back into `ACTIVE` (or another workspace)
- After merging, the workspace reference can be cleared

This approach encourages short-lived feature spaces, reduces confusion about merging older, stale branches, and fosters simpler merges.

### 5. Advanced Merge & Conflict Resolution

We provide a structural merge engine for JSON/YAML file types, plus a line-based fallback for everything else. N-way merges are supported by iteratively merging each parent's tree. Future expansions (e.g., merging XML or domain-specific file types) are straightforward to add via plugins in `internal/plugins`.

### 6. Commit Signing & Verification

Commits can be optionally signed with an Ed25519 private key. Evo can store passphrase-protected private keys in `.evo/keys/`. On `evo log`, you'll see if each commit is "verified" or not. This ensures authenticity and integrity in large or distributed teams.

### 7. Large File Support

Files exceeding a threshold (e.g., 5MB) can be moved into `.evo/largefiles` automatically, leaving a small "stub" behind in the working directory. This keeps repository sizes more manageable.

### 8. Fine-Grained Concurrency

We use a combination of in-memory locks (Go `sync.Mutex`) keyed by subdirectory or resource, ensuring multiple Evo commands or server requests can run without corrupting references. In the future, these can be replaced or augmented by file-based locking if needed.

### 9. Built-In HTTP Server

Running `evo server` launches an HTTP service for:
- Pull (`GET /pull`) to list known commits
- Push (`POST /push`) to upload missing commit objects
- Get Objects (`GET /objects/<hash>`) to fetch specific commit or tree data

This allows you to self-host Evo on any system. The server also includes stubbed endpoints (`/auth/login`, `/auth/register`) for user authentication, which can be expanded to meet enterprise needs (OAuth, LDAP, etc.).

### 10. Friendly CLI

Commands revolve around the core tasks:
```bash
evo init      # Initialize a new Evo repo
evo status    # List added, modified, deleted, or renamed files
evo commit    # Capture changes in a new commit. --sign to sign, --partial for staged-only
evo revert    # Invert a specific commit's changes
evo workspace # Manage ephemeral workspaces (create, switch, merge, list)
evo sync      # Pull from remote, handle merges if needed, then push local commits
evo log       # Show commit history in descending time order (with optional signature verification)
evo server    # Start an Evo server instance for push/pull hosting
```

## How to Contribute

### 1. Fork or Clone the Repo
- Standard GitHub flow if the project is hosted on GH. Or get Evo from the official source.

### 2. Set Up Your Environment
- You'll need Go 1.20+ installed
- Run `go build ./cmd/evo` to produce the evo binary

### 3. Branch/Workspace for Features
- Use `evo workspace create <feature>` to start new features, or rely on normal Git branching if you're just storing code in a Git repo for now

### 4. Coding Conventions
- The codebase is broken into `internal/core`, `internal/server`, `internal/commands`, etc.
- Keep your changes modular: if you're adding a new type of merge plugin, place it in `internal/plugins/`
- If you're altering the CLI, look in `internal/commands/`
- For concurrency or data structures, check `internal/core/`

### 5. Add Tests
- We encourage adding tests in a `tests/` or alongside relevant packages
- For merges, revert logic, or large files, test real-world scenarios and corner cases

### 6. Open a Pull Request (if using GitHub or a similar platform)
- Provide a brief description of your changes, referencing any open issues or planned features
- We'll review and discuss as needed

## Future Roadmap

- **More Detailed Renames**: The current rename detection is naive. We plan to implement more robust logic (like comparing file similarity thresholds) to track renames accurately
- **Improved Conflict UI**: An interactive conflict-resolution mode for merges, guiding the user through each file's conflict
- **Tree Compression / Packfiles**: Similar to Git's packfiles, for more efficient storage in huge repos
- **Extended Server Features**: Full user auth management, code reviews, integrated CI triggers, etc.
- **GUI Clients**: Although Evo is CLI-first, we welcome community-driven GUIs

## Summary

Evo aims to reinvent the developer experience of version control by focusing on clarity, simpler branching (workspaces), advanced merges, and robust offline support. Thanks to Go's concurrency features, built-in cryptographic libraries, and static binary distribution, it can serve small teams or large enterprises with minimal overhead.

We look forward to your contributions—whether that's adding new merge strategies, refining the commit process, or extending the server's capabilities. Welcome to Evo!

## Questions or Ideas?

- Please open an issue or discussion in the Evo repository
- Or jump into the code, add your feature, and open a PR
