package main

import (
	"evo/internal/commands"
	"fmt"

	"github.com/spf13/cobra"
)

var (
    serverPort   int
    serverRepo   string
)

func init() {
    var serverCmd = &cobra.Command{
        Use:   "server",
        Short: "Run Evo server for push/pull",
        Long:  `Starts Evo's built-in HTTP server to allow remote push/pull, authentication, etc.`,
        Run: func(cmd *cobra.Command, args []string) {
            // We'll mirror old 'flag' behavior
            var forwarded []string
            forwarded = append(forwarded, "--port", fmt.Sprintf("%d", serverPort))
            forwarded = append(forwarded, "--repo", serverRepo)

            commands.RunServer(forwarded)
        },
    }

    serverCmd.Flags().IntVar(&serverPort, "port", 8080, "Port to run Evo server on")
    serverCmd.Flags().StringVar(&serverRepo, "repo", "", "Path to the server's Evo repository")

    rootCmd.AddCommand(serverCmd)
}
