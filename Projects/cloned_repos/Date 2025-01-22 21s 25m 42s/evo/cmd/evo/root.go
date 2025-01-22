package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "evo",
    Short: "Evo - Next-Generation Version Control System",
    Long:  `Evo is an offline-first, tree-based version control system (VCS) with ephemeral workspaces, advanced merges, commit signing, etc.`,
    // We can define a Run here if we want something to happen when you just type `evo`,
    // but typically we let subcommands do the work.
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
