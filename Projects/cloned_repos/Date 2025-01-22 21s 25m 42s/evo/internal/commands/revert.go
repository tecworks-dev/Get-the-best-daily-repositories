package commands

import (
	"evo/internal/core"
	"fmt"
)

func RunRevert(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: evo revert <commit-hash>")
		return
	}
	hash := args[0]

	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	// We'll revert the commit by inverting its diffs
	newHash, err := core.RevertCommit(repoPath, hash)
	if err != nil {
		fmt.Println("Revert failed:", err)
		return
	}

	fmt.Println("Created revert commit:", newHash)
}
