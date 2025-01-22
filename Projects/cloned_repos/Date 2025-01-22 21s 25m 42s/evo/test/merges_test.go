package tests

import (
	"evo/internal/core"
	"os"
	"path/filepath"
	"testing"
)

func TestSimpleMergeWorkspace(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "evo-merge-test-*")
	defer os.RemoveAll(tmpDir)

	if err := core.InitRepo(tmpDir); err != nil {
		t.Fatalf("InitRepo: %v", err)
	}
	user := core.UserConfig{Name: "Merger", Email: "merge@example.com"}
	core.SaveUserConfig(tmpDir, user)

	// 1) Create base commit
	fileA := filepath.Join(tmpDir, "fileA.txt")
	os.WriteFile(fileA, []byte("base content"), 0644)
	baseHash, err := core.CreateCommit(tmpDir, "base commit", false, false, user)
	if err != nil {
		t.Fatalf("CreateCommit(base) failed: %v", err)
	}
	if baseHash == "" {
		t.Fatal("empty base hash")
	}

	// 2) Create workspace
	if err := core.CreateWorkspace(tmpDir, "feature"); err != nil {
		t.Fatalf("CreateWorkspace: %v", err)
	}
	if err := core.SwitchWorkspace(tmpDir, "feature"); err != nil {
		t.Fatalf("SwitchWorkspace: %v", err)
	}

	// 3) Modify fileA => new commit in workspace
	os.WriteFile(fileA, []byte("FEATURE content"), 0644)
	_, err = core.CreateCommit(tmpDir, "ws commit", false, false, user)
	if err != nil {
		t.Fatalf("CreateCommit(ws) failed: %v", err)
	}

	// 4) Switch back to ACTIVE (the original base commit).
	// Typically you'd do "evo workspace switch main" or something,
	// but here let's directly write the ref.
	if err := core.WriteRef(filepath.Join(tmpDir, core.EvoDir), core.ActiveRef, baseHash); err != nil {
		t.Fatalf("manually switching to baseHash: %v", err)
	}

	// 5) Merge workspace => merges in the new commits from "feature"
	err = core.MergeWorkspace(tmpDir, "feature")
	if err != nil {
		t.Fatalf("MergeWorkspace: %v", err)
	}

	// 6) Confirm .evo/workspaces/feature is cleared
	wss, _ := core.ListWorkspaces(tmpDir)
	if len(wss) != 0 {
		t.Errorf("Expected no workspaces after merge, got %v", wss)
	}

	// 7) Confirm fileA content is from the workspace => "FEATURE content"
	data, _ := os.ReadFile(fileA)
	if string(data) != "FEATURE content" {
		t.Errorf("FileA mismatch: want 'FEATURE content', got '%s'", string(data))
	}
}
