package core

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type IndexEntry struct {
    ModTime int64  `json:"modtime"`
    Hash    string `json:"hash"`
}

type EvoIndex struct {
    Files map[string]IndexEntry `json:"files"`
}

func LoadIndex(repoPath string) (*EvoIndex, error) {
    idxPath := filepath.Join(repoPath, EvoDir, "index.json")
    b, err := os.ReadFile(idxPath)
    if os.IsNotExist(err) {
        return &EvoIndex{Files: map[string]IndexEntry{}}, nil
    } else if err != nil {
        return nil, err
    }
    var idx EvoIndex
    if uerr := json.Unmarshal(b, &idx); uerr != nil {
        return &EvoIndex{Files: map[string]IndexEntry{}}, nil
    }
    return &idx, nil
}

func SaveIndex(repoPath string, idx *EvoIndex) error {
    idxPath := filepath.Join(repoPath, EvoDir, "index.json")
    data, _ := json.MarshalIndent(idx, "", "  ")
    return os.WriteFile(idxPath, data, 0644)
}

// getOrComputeHash checks if the file's modtime matches the index; if so, return the cached hash
func getOrComputeHash(repoPath, relPath string, idx *EvoIndex) (string, error) {
    absPath := filepath.Join(repoPath, relPath)
    fi, err := os.Stat(absPath)
    if err != nil {
        return "", err
    }
    mod := fi.ModTime().UnixNano()

    if entry, ok := idx.Files[relPath]; ok {
        if entry.ModTime == mod {
            // use cached
            return entry.Hash, nil
        }
    }

    // compute
    h, err := fileBlobHash(absPath)
    if err != nil {
        return "", err
    }
    idx.Files[relPath] = IndexEntry{
        ModTime: mod,
        Hash:    h,
    }
    return h, nil
}
