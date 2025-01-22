package tests

import (
	"evo/internal/core"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkspaceLifecycle(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "evo-workspace-test-*")
	defer os.RemoveAll(tmpDir)

	// Init
	if err := core.InitRepo(tmpDir); err != nil {
		t.Fatalf("InitRepo: %v", err)
	}

	// Write a file => create a commit => so that ACTIVE is not empty
	user := core.UserConfig{Name: "WsUser", Email: "ws@example.com"}
	_ = core.SaveUserConfig(tmpDir, user)
	testFile := filepath.Join(tmpDir, "wsfile.txt")
	os.WriteFile(testFile, []byte("content"), 0644)
	_, _ = core.CreateCommit(tmpDir, "Base commit", false, false, user)

	// Create a workspace
	if err := core.CreateWorkspace(tmpDir, "feature-xyz"); err != nil {
		t.Fatalf("CreateWorkspace: %v", err)
	}

	// List
	wss, err := core.ListWorkspaces(tmpDir)
	if err != nil {
		t.Fatalf("ListWorkspaces: %v", err)
	}
	if len(wss) != 1 || wss[0] != "feature-xyz" {
		t.Fatalf("Expected one workspace: feature-xyz, got: %v", wss)
	}

	// Switch
	err = core.SwitchWorkspace(tmpDir, "feature-xyz")
	if err != nil {
		t.Fatalf("SwitchWorkspace: %v", err)
	}
}
