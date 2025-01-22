package main

import (
	"evo/internal/core"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
    var switchCmd = &cobra.Command{
        Use:   "switch <name>",
        Short: "Switch to an existing workspace",
        Run: func(cmd *cobra.Command, args []string) {
            if len(args) < 1 {
                fmt.Println("Usage: evo workspace switch <name>")
                return
            }
            repoPath, err := core.FindRepoRoot(".")
            if err != nil {
                fmt.Println("Not in an Evo repository.")
                return
            }
            if err := core.SwitchWorkspace(repoPath, args[0]); err != nil {
                fmt.Println("Error switching workspace:", err)
                return
            }
            fmt.Println("Switched to workspace:", args[0])
        },
    }

    workspaceCmd.AddCommand(switchCmd)
}
