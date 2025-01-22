package main

import (
	"evo/internal/commands"

	"github.com/spf13/cobra"
)

func init() {
    var logCmd = &cobra.Command{
        Use:   "log",
        Short: "Show commit history",
        Long:  `Displays a list of commits in descending time order.`,
        Run: func(cmd *cobra.Command, args []string) {
            commands.RunLog(args)
        },
    }
    rootCmd.AddCommand(logCmd)
}
