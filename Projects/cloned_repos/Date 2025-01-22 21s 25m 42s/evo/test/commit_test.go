package tests

import (
	"evo/internal/core"
	"os"
	"path/filepath"
	"testing"
)

func TestCreateCommit_EmptyRepo(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "evo-commit-test-*")
	defer os.RemoveAll(tmpDir)

	// Initialize an empty Evo repo
	if err := core.InitRepo(tmpDir); err != nil {
		t.Fatalf("init repo failed: %v", err)
	}

	// Make a sample file
	testFile := filepath.Join(tmpDir, "hello.txt")
	os.WriteFile(testFile, []byte("Hello Evo!"), 0644)

	user := core.UserConfig{Name: "Test User", Email: "test@example.com"}
	_ = core.SaveUserConfig(tmpDir, user) // not strictly necessary, but typical usage

	// Create a commit
	hash, err := core.CreateCommit(tmpDir, "Initial commit", false, false, user)
	if err != nil {
		t.Fatalf("CreateCommit failed: %v", err)
	}
	if hash == "" {
		t.Fatal("Expected a commit hash, got empty string")
	}
}

func TestCreateCommit_Partial(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "evo-commit-partial-*")
	defer os.RemoveAll(tmpDir)

	if err := core.InitRepo(tmpDir); err != nil {
		t.Fatalf("InitRepo: %v", err)
	}

	// Write 2 files, only 1 will be staged
	fileA := filepath.Join(tmpDir, "fileA.txt")
	os.WriteFile(fileA, []byte("A content"), 0644)
	fileB := filepath.Join(tmpDir, "fileB.txt")
	os.WriteFile(fileB, []byte("B content"), 0644)

	user := core.UserConfig{Name: "Partial", Email: "partial@example.com"}
	core.SaveUserConfig(tmpDir, user)

	// Stage fileA only
	stagingDir := filepath.Join(tmpDir, core.EvoDir, "staging")
	os.MkdirAll(stagingDir, 0755)
	indexPath := filepath.Join(stagingDir, "index")
	os.WriteFile(indexPath, []byte("fileA.txt\n"), 0644)

	// partial = true => only commits fileA
	hash, err := core.CreateCommit(tmpDir, "Partial commit", false, true, user)
	if err != nil {
		t.Fatalf("partial CreateCommit failed: %v", err)
	}
	if hash == "" {
		t.Fatal("Expected commit hash, got empty")
	}

	// The new commit should have exactly 1 file in its tree
	evoPath := filepath.Join(tmpDir, core.EvoDir)
	c, err := core.ReadCommitObject(evoPath, hash)
	if err != nil {
		t.Fatalf("ReadCommitObject failed: %v", err)
	}
	if c.TreeHash == "" {
		t.Fatalf("commit has no treeHash")
	}

	tObj, err := core.ReadTreeObject(evoPath, c.TreeHash)
	if err != nil {
		t.Fatalf("ReadTreeObject failed: %v", err)
	}
	if len(tObj.Files) != 1 {
		t.Errorf("expected 1 file in commit, got %d", len(tObj.Files))
	}
	if _, ok := tObj.Files["fileA.txt"]; !ok {
		t.Errorf("partial commit missing fileA, or has different files than expected")
	}
}
