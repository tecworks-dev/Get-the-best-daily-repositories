package main

import (
	"evo/internal/core"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
    var listCmd = &cobra.Command{
        Use:   "list",
        Short: "List all workspaces",
        Run: func(cmd *cobra.Command, args []string) {
            repoPath, err := core.FindRepoRoot(".")
            if err != nil {
                fmt.Println("Not in an Evo repository.")
                return
            }
            wss, err := core.ListWorkspaces(repoPath)
            if err != nil {
                fmt.Println("Error listing workspaces:", err)
                return
            }
            for _, w := range wss {
                fmt.Println(w)
            }
        },
    }
    workspaceCmd.AddCommand(listCmd)
}
