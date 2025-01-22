# Evo: The Evolution of Version Control

[![Go Report Card](https://goreportcard.com/badge/github.com/crazywolf132/evo)](https://goreportcard.com/report/github.com/crazywolf132/evo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Go Version](https://img.shields.io/github/go-mod/go-version/crazywolf132/evo)](https://golang.org/)

> Version control that works the way you think

## Why Evo?

Ever felt that version control should be simpler? That branching shouldn't require a PhD in Git? That merging shouldn't feel like defusing a bomb? We did too. That's why we built Evo.

Evo is a modern, offline-first version control system that focuses on what matters most: helping developers write great code together. No more merge conflicts that make you want to quit programming. No more branch structures that look like abstract art. Just clean, intuitive version control that works.

## ✨ Features That Make a Difference

### 🌿 Workspaces, Not Branches
```bash
evo workspace create feature-auth
# Make your changes, commit, and when ready...
evo workspace merge
```
Forget long-lived branches. Workspaces are ephemeral development environments that encourage clean, focused work. Create one for each feature, merge it when you're done, and move on. Simple.

### 🤝 Smart Merges That Actually Work
- **Structural merging** for JSON and YAML files
- **Intelligent conflict resolution** that understands your code
- **N-way merges** when you need them
- No more "Accept Current" or "Accept Incoming" blind choices

### 🔒 Built for Teams
- **Offline-first**: Work anywhere, sync when ready
- **Fine-grained concurrency**: Multiple team members can work on different parts of the codebase simultaneously
- **Built-in auth**: Simple user management with extensible authentication
- **Optional commit signing**: Verify who wrote what with Ed25519

### 📦 Enterprise Ready
- **Large file support** built-in
- **Structural merges** for configuration files
- **Fine-grained permissions**
- **Self-hosted option** with built-in HTTP server

## 🚀 Quick Start

```bash
# Install Evo
go install github.com/crazywolf132/evo/cmd/evo@latest

# Initialize a new repository
evo init

# Create a workspace for your feature
evo workspace create my-feature

# Stage and commit your changes
evo commit -m "Add awesome feature"

# Merge your changes when ready
evo workspace merge

# Push to a remote server
evo sync
```

## 🎯 Core Philosophy

Evo was built on three core principles:

1. **Simplicity is power**: Complex features should have simple interfaces
2. **Work flows like thought**: Version control should match your mental model
3. **Trust through verification**: Every commit can be traced and verified

## 🛠 Built with Modern Tech

- **Go**: For reliability, performance, and ease of deployment
- **Ed25519**: For secure commit signing
- **HTTP/2**: For efficient client-server communication
- **Structural Parsing**: For intelligent merges

## 📚 Documentation

- [Getting Started Guide](./docs/getting-started.md)
- [Command Reference](./docs/commands.md)
- [Design Document](./DESIGN.md)
- [Contributing Guide](./CONTRIBUTING.md)

## 🤝 Contributing

We believe great tools come from great communities. Whether you're fixing bugs, improving documentation, or adding new features, we'd love your help! See our [Contributing Guide](./CONTRIBUTING.md) to get started.

## 📜 License

Evo is open source software licensed under the MIT license. See the [LICENSE](./LICENSE) file for more details.

---

Built with ❤️ by developers, for developers.

Looking to migrate from another VCS? Check out our [migration guides](./docs/migrations/).
