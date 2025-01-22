package main

import (
	"evo/internal/commands"

	"github.com/spf13/cobra"
)

func init() {
    // define a new subcommand
    var initCmd = &cobra.Command{
        Use:   "init [directory]",
        Short: "Initialize an Evo repository",
        Long: `Initialize a new Evo repository in the specified directory.
If no directory is provided, the current directory is used.`,
        Run: func(cmd *cobra.Command, args []string) {
            var path string
            if len(args) > 0 {
                path = args[0]
            } else {
                path = "."
            }
            commands.RunInit([]string{path})
        },
    }

    // attach it to the root command
    rootCmd.AddCommand(initCmd)
}
