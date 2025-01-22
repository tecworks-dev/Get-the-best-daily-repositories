package main

import (
	"evo/internal/core"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
    var mergeCmd = &cobra.Command{
        Use:   "merge <name>",
        Short: "Merge a workspace into the current ACTIVE commit",
        Run: func(cmd *cobra.Command, args []string) {
            if len(args) < 1 {
                fmt.Println("Usage: evo workspace merge <name>")
                return
            }
            repoPath, err := core.FindRepoRoot(".")
            if err != nil {
                fmt.Println("Not in an Evo repository.")
                return
            }
            if err := core.MergeWorkspace(repoPath, args[0]); err != nil {
                fmt.Println("Error merging workspace:", err)
                return
            }
            fmt.Println("Workspace merged:", args[0])
        },
    }

    workspaceCmd.AddCommand(mergeCmd)
}
