package core

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

const (
	EvoDir    = ".evo"
	ActiveRef = "ACTIVE" // Replace "HEAD" with "ACTIVE" for Evo's current commit.
)

// for concurrency safety, each repo can maintain a global lock.
// In a real system, you might have fine-grained locks or use file locking.
var repoMutexes = struct {
	sync.Mutex // map[string]*sync.Mutex keyed by absolute repo path
}{}

// InitRepo creates the .evo structure, sets up initial references, config, etc.
func InitRepo(path string) error {
	evoPath := filepath.Join(path, EvoDir)
	if _, err := os.Stat(evoPath); err == nil {
		return fmt.Errorf("Evo repository already exists at %s", evoPath)
	}

	// Create subdirs
	dirs := []string{
		"objects",
		"refs",
		"workspaces",
		"largefiles",
		"staging",
		"keys",
		"config",
	}
	for _, d := range dirs {
		err := os.MkdirAll(filepath.Join(evoPath, d), 0755)
		if err != nil {
			return err
		}
	}

	// Initialize HEAD to empty
	err := WriteRef(evoPath, ActiveRef, "")
	if err != nil {
		return err
	}

	// Optionally set a default remote.
	return nil
}

func FindRepoRoot(start string) (string, error) {
	abs, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}

	for {
		evoPath := filepath.Join(abs, EvoDir)
		fi, err := os.Stat(evoPath)
		if err == nil && fi.IsDir() {
			return abs, nil
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			return "", errors.New("no .evo found in any parent directory")
		}
		abs = parent
	}
}
