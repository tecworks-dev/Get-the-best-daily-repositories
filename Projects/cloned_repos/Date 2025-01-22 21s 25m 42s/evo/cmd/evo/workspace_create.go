package main

import (
	"evo/internal/core"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
	var createCmd = &cobra.Command{
		Use:   "create <name>",
		Short: "Create a new workspace",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) < 1 {
				fmt.Println("Usage: evo workspace create <name>")
				return
			}
			repoPath, err := core.FindRepoRoot(".")
			if err != nil {
				fmt.Println("Not in an Evo repository.")
				return
			}
			if err := core.CreateWorkspace(repoPath, args[0]); err != nil {
				fmt.Println("Error creating workspace:", err)
				return
			}
			fmt.Println("Workspace created:", args[0])
		},
	}

	workspaceCmd.AddCommand(createCmd)
}
