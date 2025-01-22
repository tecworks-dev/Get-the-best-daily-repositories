// HTTP handlers for pull, push, getObjects, etc.
package server

import (
	"encoding/base64"
	"encoding/json"
	"evo/internal/core"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func handlePull(w http.ResponseWriter, r *http.Request) {
    repoPath, err := getServerRepoPath()
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    commits, err := core.GetCommitLog(repoPath)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    var hashes []string
    for _, c := range commits {
        hashes = append(hashes, c.Hash)
    }
    data, _ := json.Marshal(hashes)
    w.Header().Set("Content-Type", "application/json")
    w.Write(data)
}

func handlePush(w http.ResponseWriter, r *http.Request) {
    repoPath, err := getServerRepoPath()
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    var obj struct {
        Hash string `json:"hash"`
        Data string `json:"data"` // base64 encoded object
    }
    body, _ := io.ReadAll(r.Body)
    if err := json.Unmarshal(body, &obj); err != nil {
        http.Error(w, "invalid json", http.StatusBadRequest)
        return
    }

    evoPath := filepath.Join(repoPath, core.EvoDir, "objects")
    outPath := filepath.Join(evoPath, obj.Hash+".json")
    if _, err := os.Stat(outPath); os.IsNotExist(err) {
        decoded, err := base64.StdEncoding.DecodeString(obj.Data)
        if err != nil {
            http.Error(w, "base64 decode error", http.StatusBadRequest)
            return
        }
        if err := os.WriteFile(outPath, decoded, 0644); err != nil {
            http.Error(w, "cannot write object", http.StatusInternalServerError)
            return
        }
    }

    w.WriteHeader(http.StatusOK)
    w.Write([]byte("OK"))
}

func handleGetObject(w http.ResponseWriter, r *http.Request) {
    repoPath, err := getServerRepoPath()
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    parts := strings.Split(r.URL.Path, "/")
    if len(parts) < 3 {
        http.NotFound(w, r)
        return
    }
    hash := parts[len(parts)-1]
    evoPath := filepath.Join(repoPath, core.EvoDir, "objects", hash+".json")
    if _, err := os.Stat(evoPath); os.IsNotExist(err) {
        http.NotFound(w, r)
        return
    }
    data, _ := os.ReadFile(evoPath)
    w.Header().Set("Content-Type", "application/json")
    w.Write(data)
}

// Basic auth endpoints (stubbed)
func handleLogin(w http.ResponseWriter, r *http.Request) {
    // Check user credentials, return token
    w.Write([]byte(`{"token":"dummy-token"}`))
}

func handleRegister(w http.ResponseWriter, r *http.Request) {
    // Create new user in server
    w.WriteHeader(http.StatusCreated)
}
