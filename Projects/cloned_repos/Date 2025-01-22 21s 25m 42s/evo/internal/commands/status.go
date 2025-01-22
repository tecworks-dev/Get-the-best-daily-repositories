package commands

import (
	"evo/internal/core"
	"fmt"

	"github.com/fatih/color"
)

func RunStatus(args []string) {
	repoPath, err := core.FindRepoRoot(".")
	if err != nil {
		fmt.Println("Not in an Evo repository.")
		return
	}

	wc, err := core.GetWorkingChanges(repoPath)
	if err != nil {
		fmt.Println("Error reading changes:", err)
		return
	}

	// If no changes:
	if len(wc.Added) == 0 && len(wc.Modified) == 0 && len(wc.Deleted) == 0 && len(wc.Renamed) == 0 {
		color.New(color.FgGreen).Println("No changes.")
		return
	}

	addedColor := color.New(color.FgGreen)
	modColor := color.New(color.FgYellow)
	delColor := color.New(color.FgRed)
	renColor := color.New(color.FgCyan)

	if len(wc.Added) > 0 {
		addedColor.Println("Added:")
		for _, f := range wc.Added {
			addedColor.Println("  ", f)
		}
	}
	if len(wc.Modified) > 0 {
		modColor.Println("Modified:")
		for _, f := range wc.Modified {
			modColor.Println("  ", f)
		}
	}
	if len(wc.Deleted) > 0 {
		delColor.Println("Deleted:")
		for _, f := range wc.Deleted {
			delColor.Println("  ", f)
		}
	}
	if len(wc.Renamed) > 0 {
		renColor.Println("Renamed:")
		for _, r := range wc.Renamed {
			renColor.Printf("  %s -> %s\n", r.OldPath, r.NewPath)
		}
	}

	fmt.Printf("\n%d added, %d modified, %d deleted, %d renamed\n", len(wc.Added), len(wc.Modified), len(wc.Deleted), len(wc.Renamed))
}
