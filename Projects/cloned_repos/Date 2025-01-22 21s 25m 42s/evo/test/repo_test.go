package tests

import (
	"evo/internal/core"
	"os"
	"path/filepath"
	"testing"
)

func TestInitRepo(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "evo-init-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 1) Init the repository
	err = core.InitRepo(tmpDir)
	if err != nil {
		t.Fatalf("InitRepo() failed: %v", err)
	}

	// 2) Check that .evo exists
	evoPath := filepath.Join(tmpDir, core.EvoDir)
	if _, err := os.Stat(evoPath); os.IsNotExist(err) {
		t.Fatalf("missing .evo folder after init")
	}

	// 3) Verify we can detect the repo root
	root, err := core.FindRepoRoot(tmpDir)
	if err != nil {
		t.Fatalf("FindRepoRoot() error: %v", err)
	}
	if root != tmpDir {
		t.Fatalf("FindRepoRoot() returned %s, want %s", root, tmpDir)
	}
}

func TestInitRepo_AlreadyExists(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "evo-init-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 1) Init once
	if err := core.InitRepo(tmpDir); err != nil {
		t.Fatalf("InitRepo() the first time: %v", err)
	}

	// 2) Attempt to init again => should return an error
	err = core.InitRepo(tmpDir)
	if err == nil {
		t.Fatal("Expected an error when initializing an already-existing repo, got nil")
	}
}
