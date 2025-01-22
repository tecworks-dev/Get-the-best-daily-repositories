package commands

import (
	"evo/internal/core"
	"fmt"
)

func RunLog(args []string) {
	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	commits, err := core.GetCommitLog(repoPath)
	if err != nil {
		fmt.Println("Error retrieving commit log:", err)
		return
	}

	// By design, they're returning in time-desc order
	for _, c := range commits {
		shortHash := c.Hash
		if len(shortHash) > 7 {
			shortHash = shortHash[:7]
		}
		verifiedStr := ""
		if c.Signature != "" {
			if core.VerifyCommit(repoPath, &c) {
				verifiedStr = " (verified)"
			} else {
				verifiedStr = " (signature mismatch!)"
			}
		}
		fmt.Printf("commit %s%s\nAuthor: %s\nDate:   %s\n\n    %s\n\n",
			shortHash,
			verifiedStr,
			c.Author,
			c.Timestamp.Local(),
			c.Message,
		)
	}
}
