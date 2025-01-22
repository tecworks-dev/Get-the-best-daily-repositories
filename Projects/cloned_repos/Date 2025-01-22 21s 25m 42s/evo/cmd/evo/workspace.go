package main

import (
	"github.com/spf13/cobra"
)

// workspaceCmd is the parent command
var workspaceCmd = &cobra.Command{
    Use:   "workspace",
    Short: "Manage ephemeral workspaces",
    Long:  `Create, merge, list, or switch ephemeral Evo workspaces.`,
}

func init() {
    // We'll add subcommands (create, merge, list, switch) to workspaceCmd below.
    rootCmd.AddCommand(workspaceCmd)
}
