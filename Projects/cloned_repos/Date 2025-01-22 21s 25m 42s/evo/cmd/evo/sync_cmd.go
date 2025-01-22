package main

import (
	"evo/internal/commands"

	"github.com/spf13/cobra"
)

func init() {
    var syncCmd = &cobra.Command{
        Use:   "sync [remote-url]",
        Short: "Synchronize local changes with a remote server",
        Long: `Pull from the remote, handle merges, and then push local commits
to keep everything in sync with the specified or default remote.`,
        Run: func(cmd *cobra.Command, args []string) {
            commands.RunSync(args)
        },
    }
    rootCmd.AddCommand(syncCmd)
}
