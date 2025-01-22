package commands

import (
	"evo/internal/core"
	"fmt"
)

func RunSync(args []string) {
	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	remoteURL := ""
	if len(args) > 0 {
		remoteURL = args[0]
	}
	if remoteURL == "" {
		// read from config or "origin"
		remoteURL, _ = core.GetRemoteURL(repoPath)
	}
	// Check if it is still empty.
	if remoteURL == "" {
		fmt.Println("No remote specified. Usage: evo sync <remote-url>")
		return
	}

	err = core.Pull(repoPath, remoteURL)
	if err != nil {
		fmt.Println("Pull failed: ", err)
		return
	}

	// merged any fetched commits if needed.
	err = core.Push(repoPath, remoteURL)
	if err != nil {
		fmt.Println("Push failed:", err)
		return
	}

	fmt.Println("Sync completed with remote:", remoteURL)
}
