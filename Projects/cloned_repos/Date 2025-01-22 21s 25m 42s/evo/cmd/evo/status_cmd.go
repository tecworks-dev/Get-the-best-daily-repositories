package main

import (
	"evo/internal/commands"

	"github.com/spf13/cobra"
)

func init() {
    var statusCmd = &cobra.Command{
        Use:   "status",
        Short: "Show working-directory changes",
        Long:  `Displays added, modified, deleted, or renamed files in the current Evo repository.`,
        Run: func(cmd *cobra.Command, args []string) {
            commands.RunStatus(args)
        },
    }
    rootCmd.AddCommand(statusCmd)
}
