package commands

import (
	"evo/internal/server"
	"flag"
	"fmt"
	"os"
)

func RunServer(args []string) {
	fs := flag.NewFlagSet("server", flag.ExitOnError)
	port := fs.Int("port", 8080, "Port to run Evo server on")
	repoPath := fs.String("repo", "", "Path to the server's Evo repository")
	fs.Parse(args)

	if *repoPath == "" {
		fmt.Println("Please specify --repo for the server's storage.")
		os.Exit(1)
	}

	os.Setenv("EVO_SERVER_REPO", *repoPath)
	srv := server.NewEvoHTTPServer(*port)
	fmt.Printf("Starting Evo server on port %d with repo: %s\n", *port, *repoPath)
	err := srv.ListenAndServe()
	if err != nil {
		fmt.Println("Server error:", err)
	}
}
