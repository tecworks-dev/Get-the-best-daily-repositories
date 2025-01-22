package main

import (
	"evo/internal/commands"
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
	var conflictsCmd = &cobra.Command{
		Use:   "conflicts",
		Short: "Manage merge conflicts in Evo",
		Long: `Provides commands for listing or resolving merge conflicts that occur
when merges or rebases produce conflicting changes in the working tree.`,
	}

	var listCmd = &cobra.Command{
		Use:   "list",
		Short: "List all files currently in conflict",
		Run: func(cmd *cobra.Command, args []string) {
			commands.RunConflictsList(args)
		},
	}

	var resolveCmd = &cobra.Command{
		Use:   "resolve <file>",
		Short: "Mark a conflict as resolved once you've fixed the file",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) < 1 {
				fmt.Println("Usage: evo conflicts resolve <file>")
				return
			}
			commands.RunConflictsResolve(args[0])
		},
	}

	conflictsCmd.AddCommand(listCmd, resolveCmd)
	rootCmd.AddCommand(conflictsCmd)
}
