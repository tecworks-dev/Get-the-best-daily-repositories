package core

import (
	"fmt"
	"os"
	"path/filepath"
)

// CreateWorkspace sets a new ref under workspaces/<name> from ACTIVE
func CreateWorkspace(repoPath, name string) error {
	evoPath := filepath.Join(repoPath, EvoDir)
	active, _ := ReadRef(evoPath, ActiveRef)
	if active == "" {
		return fmt.Errorf("no ACTIVE commit to base from")
	}
	// Write a new ref at .evo/workspaces/<name> pointing to the current ACTIVE
	return WriteRef(evoPath, "workspaces/"+name, active)
}

// ListWorkspaces lists all ephemeral workspace references in .evo/workspaces/
func ListWorkspaces(repoPath string) ([]string, error) {
	evoPath := filepath.Join(repoPath, EvoDir, "workspaces")
	entries, err := os.ReadDir(evoPath)
	if os.IsNotExist(err) {
		// If there is no .evo/workspaces folder, return empty
		return []string{}, nil
	}
	if err != nil {
		return nil, err
	}

	var wss []string
	for _, e := range entries {
		// Each ref is just a file
		if !e.IsDir() {
			wss = append(wss, e.Name())
		}
	}
	return wss, nil
}

// SwitchWorkspace sets ACTIVE to the commit pointed to by .evo/workspaces/<name>
func SwitchWorkspace(repoPath, name string) error {
	evoPath := filepath.Join(repoPath, EvoDir)
	wsRef := "workspaces/" + name
	wsActive, err := ReadRef(evoPath, wsRef)
	if err != nil || wsActive == "" {
		return fmt.Errorf("workspace '%s' not found (or empty)", name)
	}
	return WriteRef(evoPath, ActiveRef, wsActive)
}
