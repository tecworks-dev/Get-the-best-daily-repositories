package commands

import (
	"evo/internal/core"
	"fmt"
	"path/filepath"
)

func RunInit(args []string) {
	path := "."
	if len(args) > 0 {
		path = args[0]
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		fmt.Println("Error resolving path:", err)
		return
	}

	err = core.InitRepo(abs)
	if err != nil {
		fmt.Println("Failed to init repo:", err)
		return
	}

	fmt.Println("Initialized Evo repository at", filepath.Join(abs, core.EvoDir))

}
