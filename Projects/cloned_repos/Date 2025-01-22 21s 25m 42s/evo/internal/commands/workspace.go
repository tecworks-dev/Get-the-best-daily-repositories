package commands

import (
	"evo/internal/core"
	"fmt"
)

func RunWorkspace(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: evo workspace <create|merge|list|switch> [name]")
		return
	}

	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	action := args[0]
	switch action {
	case "create":
		if len(args) < 2 {
			fmt.Println("Usage: evo workspace create <name>")
			return
		}
		name := args[1]
		err := core.CreateWorkspace(repoPath, name)
		if err != nil {
			fmt.Println("Error creating workspace:", err)
			return
		}
		fmt.Println("Workspace created:", name)

	case "merge":
		if len(args) < 2 {
			fmt.Println("Usage: evo workspace merge <name>")
			return
		}
		name := args[1]
		err := core.MergeWorkspace(repoPath, name)
		if err != nil {
			fmt.Println("Error merging workspace:", err)
			return
		}
		fmt.Println("Workspace merged:", name)

	case "list":
		list, err := core.ListWorkspaces(repoPath)
		if err != nil {
			fmt.Println("Error listing workspaces:", err)
			return
		}
		for _, w := range list {
			fmt.Println(w)
		}

	case "switch":
		if len(args) < 2 {
			fmt.Println("Usage: evo workspace switch <name>")
			return
		}
		err := core.SwitchWorkspace(repoPath, args[1])
		if err != nil {
			fmt.Println("Error switching workspace:", err)
			return
		}
		fmt.Println("Switched to workspace:", args[1])

	default:
		fmt.Println("Unknown workspace command:", action)
	}
}
