package main

import (
	"evo/internal/commands"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
    var revertCmd = &cobra.Command{
        Use:   "revert <commit-hash>",
        Short: "Revert a specific commit by inverting its diffs",
        Long:  `Reverts the changes introduced by the specified commit. Creates a new commit with the inverted changes.`,
        Run: func(cmd *cobra.Command, args []string) {
            if len(args) < 1 {
                fmt.Println("Usage: evo revert <commit-hash>")
                return
            }
            commands.RunRevert(args)
        },
    }
    rootCmd.AddCommand(revertCmd)
}
