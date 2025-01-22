package commands

import (
	"evo/internal/core"
	"fmt"
	"os"
	"path/filepath"
)

// RunConflictsList lists all files in .evo/conflicts/
func RunConflictsList(args []string) {
    repoPath, err := core.FindRepoRoot(".")
    if err != nil {
        fmt.Println("Not in an Evo repository.")
        return
    }
    conflictsDir := filepath.Join(repoPath, core.EvoDir, "conflicts")
    entries, err := os.ReadDir(conflictsDir)
    if os.IsNotExist(err) {
        fmt.Println("No conflicts directory found. No active conflicts.")
        return
    } else if err != nil {
        fmt.Println("Error reading conflicts directory:", err)
        return
    }

    if len(entries) == 0 {
        fmt.Println("No conflicts found.")
        return
    }

    fmt.Println("Conflicts:")
    for _, e := range entries {
        if !e.IsDir() {
            fmt.Println("  ", e.Name())
        }
    }
}

// RunConflictsResolve marks a file as resolved
func RunConflictsResolve(file string) {
    repoPath, err := core.FindRepoRoot(".")
    if err != nil {
        fmt.Println("Not in an Evo repository.")
        return
    }
    conflictsDir := filepath.Join(repoPath, core.EvoDir, "conflicts")
    conflictFile := filepath.Join(conflictsDir, file)

    if _, err := os.Stat(conflictFile); os.IsNotExist(err) {
        fmt.Printf("No conflict file found for '%s'. Are you sure it's in conflict?\n", file)
        return
    }

    // Remove the conflict marker
    os.Remove(conflictFile)
    fmt.Printf("Marked '%s' as resolved. Please verify the file is correct, then commit.\n", file)
}
