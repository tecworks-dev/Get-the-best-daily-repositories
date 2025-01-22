package main

import (
	"evo/internal/commands"

	"github.com/spf13/cobra"
)

var (
	commitMessage string
	commitSign    bool
	commitPartial bool
)

func init() {
	var commitCmd = &cobra.Command{
		Use:   "commit",
		Short: "Create a new commit",
		Long: `Record changes in the local repository.
        You can specify a commit message, sign the commit, or do a partial commit.

        If you use --partial, you need to have staged files via 'evo stage <files>'.
        Only those staged files will be committed.`,
		Run: func(cmd *cobra.Command, args []string) {
			// Build up the arguments as if we used the old flags
			var forwarded []string
			if commitMessage != "" {
				forwarded = append(forwarded, "-m", commitMessage)
			}
			if commitSign {
				forwarded = append(forwarded, "--sign")
			}
			if commitPartial {
				forwarded = append(forwarded, "--partial")
			}
			commands.RunCommit(forwarded)
		},
	}

	// define flags on commitCmd
	commitCmd.Flags().StringVarP(&commitMessage, "message", "m", "", "Commit message")
	commitCmd.Flags().BoolVar(&commitSign, "sign", false, "Sign the commit with a local keypair")
	commitCmd.Flags().BoolVar(&commitPartial, "partial", false, "Only commit changes since the last commit")

	rootCmd.AddCommand(commitCmd)
}
