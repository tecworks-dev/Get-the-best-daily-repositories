package commands

import (
	"evo/internal/core"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func RunCommit(args []string) {
	fs := flag.NewFlagSet("commit", flag.ExitOnError)
	msg := fs.String("m", "", "Commit message")
	sign := fs.Bool("sign", false, "Sign the commit with a local keypair")
	partial := fs.Bool("partial", false, "Only commit changes since the last commit")
	fs.Parse(args)

	if strings.TrimSpace(*msg) == "" {
		fmt.Println("Please specify a commit message with -m.")
		return
	}

	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	var staged []string

	// If partial is requested, ensure user staged cahnges beforehand (we have a seperate staging mechanism).
	if *partial {
		var err error
		staged, err = core.GetStagedChanges(repoPath)
		if err != nil {
			fmt.Println("Error reading staged changes:", err)
			return
		}

		if len(staged) == 0 {
			fmt.Println("No staged changes found for partial commit.")
			return
		}
	}

	// Get current user info from config (user.name, user.email)
	user, err := core.LoadUserConfig(repoPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: no user config found, defaulting to 'Evo Dev'.\n")
	}

	commitHash, err := core.CreateCommit(repoPath, *msg, *sign, *partial, user)
	if err != nil {
		fmt.Println("Commit failed:", err)
		return
	}

	shortHash := commitHash
	if len(shortHash) > 7 {
		shortHash = shortHash[:7]
	}
	fmt.Printf("[%s] %s\n", shortHash, *msg)

	// Cleaning up the partial commit
	if *partial {
		cleanupStaging(repoPath, staged)
	}
}

func cleanupStaging(repoPath string, staged []string) {
	stPath := filepath.Join(repoPath, core.EvoDir, "staging", "index")
	data, _ := os.ReadFile(stPath)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	// remove lines that match staged
	finalLines := []string{}
	stagedSet := make(map[string]bool)
	for _, s := range staged {
		stagedSet[s] = true
	}
	for _, line := range lines {
		if !stagedSet[line] && line != "" {
			finalLines = append(finalLines, line)
		}
	}
	os.WriteFile(stPath, []byte(strings.Join(finalLines, "\n")+"\n"), 0644)
}
