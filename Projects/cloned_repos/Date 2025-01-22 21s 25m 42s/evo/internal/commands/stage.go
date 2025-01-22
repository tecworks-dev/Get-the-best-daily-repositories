package commands

import (
	"evo/internal/core"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func RunStage(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: evo stage <file...>")
		return
	}

	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	stPath := filepath.Join(repoPath, core.EvoDir, "staging", "index")

	// read existing
	data, _ := os.ReadFile(stPath)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	existing := make(map[string]bool)
	for _, l := range lines {
		if l != "" {
			existing[l] = true
		}
	}

	// add new
	for _, file := range args {
		existing[file] = true
	}

	// rewrite file
	var out []string
	for f := range existing {
		out = append(out, f)
	}
	final := strings.Join(out, "\n") + "\n"
	os.WriteFile(stPath, []byte(final), 0644)

	fmt.Printf("Staged %d files.\n", len(args))
}
