// Detect changes in working directory, handle partial staging logic, structural merges, etc.
package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// FileChanges holds sets of changed files
type FileChanges struct {
	Added    []string
	Modified []string
	Deleted  []string
	Renamed  []RenameInfo
}

// RenameInfo describes a file rename or move (old -> new).
type RenameInfo struct {
	OldPath string
	NewPath string
}

// buildTreeFromWorking compares the working directory to the last commit's tree (or empty)
// to produce a new Tree object. If `partial == true`, we only commit files listed in staging.
func buildTreeFromWorking(repoPath string, partial bool) (string, error) {
	evoPath := filepath.Join(repoPath, EvoDir)
	activeRef, _ := ReadRef(evoPath, ActiveRef)

	var baseTree *Tree
	if activeRef != "" {
		c, err := ReadCommitObject(evoPath, activeRef)
		if err != nil {
			return "", err
		}
		bt, err := ReadTreeObject(evoPath, c.TreeHash)
		if err != nil {
			return "", err
		}
		baseTree = bt
	} else {
		baseTree = &Tree{Files: make(map[string]string)}
	}

	// Clone from the base so we can edit it
	newTree := CloneTree(baseTree)

	// If partial, we only commit files in staging
	if partial {
		staged, err := GetStagedChanges(repoPath)
		if err != nil {
			return "", err
		}
		for _, st := range staged {
			absPath := filepath.Join(repoPath, st)
			if _, err := os.Stat(absPath); os.IsNotExist(err) {
				// file removed => remove from newTree
				delete(newTree.Files, st)
			} else {
				blobHash, err := storeBlobWithIndex(repoPath, st)
				if err != nil {
					return "", err
				}
				newTree.Files[st] = blobHash
			}
		}
		// No rename detection for partial commits
	} else {
		// Full commit => we compare everything to the base tree
		ignorePatterns, err := LoadIgnorePatterns(repoPath)
		if err != nil {
			return "", fmt.Errorf("failed to load evo-ignore: %v", err)
		}

		// We'll gather an index for performance so we only re-hash changed files
		idx, err := LoadIndex(repoPath)
		if err != nil {
			return "", fmt.Errorf("failed to load index: %v", err)
		}

		// Walk the entire repo, skipping .evo and ignored patterns
		wdFiles := make(map[string]bool)
		filepath.Walk(repoPath, func(p string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				if p == evoPath {
					return filepath.SkipDir
				}
				return nil
			}
			rel, _ := filepath.Rel(repoPath, p)
			if strings.HasPrefix(rel, EvoDir) {
				return nil
			}
			if MatchesIgnorePatterns(rel, ignorePatterns) {
				return nil
			}
			wdFiles[rel] = true
			return nil
		})

		changes := &FileChanges{}
		// detect new or modified
		for rel := range wdFiles {
			blobHash, err := getOrComputeHash(repoPath, rel, idx)
			if err != nil {
				return "", fmt.Errorf("failed hashing %s: %v", rel, err)
			}
			baseBlob, ok := newTree.Files[rel]
			if !ok {
				changes.Added = append(changes.Added, rel)
				newTree.Files[rel] = blobHash
			} else if baseBlob != blobHash {
				changes.Modified = append(changes.Modified, rel)
				newTree.Files[rel] = blobHash
			}
		}
		// detect deleted
		for f := range newTree.Files {
			if !wdFiles[f] {
				changes.Deleted = append(changes.Deleted, f)
				delete(newTree.Files, f)
			}
		}

		// detect renames
		detectRenames(changes, baseTree, newTree)

		// write out updated index
		if err := SaveIndex(repoPath, idx); err != nil {
			return "", err
		}
	}

	// Write the new tree object
	return writeTreeObject(evoPath, newTree)
}

// detectRenames identifies files that appear in Deleted + Added with the same content hash => rename.
func detectRenames(changes *FileChanges, baseTree, newTree *Tree) {
	deletedFiles := make(map[string]string) // blobHash -> oldPath
	for _, oldPath := range changes.Deleted {
		oldHash := baseTree.Files[oldPath]
		deletedFiles[oldHash] = oldPath
	}
	addedFiles := make(map[string]string) // blobHash -> newPath
	for _, newPath := range changes.Added {
		newHash := newTree.Files[newPath]
		addedFiles[newHash] = newPath
	}

	var renameInfos []RenameInfo
	for blobHash, oldPath := range deletedFiles {
		if newPath, ok := addedFiles[blobHash]; ok {
			// same content => rename
			renameInfos = append(renameInfos, RenameInfo{
				OldPath: oldPath,
				NewPath: newPath,
			})
		}
	}

	// filter out these from Deletions/Additions
	renamedOld := make(map[string]bool)
	renamedNew := make(map[string]bool)
	for _, ri := range renameInfos {
		renamedOld[ri.OldPath] = true
		renamedNew[ri.NewPath] = true
	}

	filteredDeleted := make([]string, 0, len(changes.Deleted))
	for _, d := range changes.Deleted {
		if !renamedOld[d] {
			filteredDeleted = append(filteredDeleted, d)
		}
	}
	changes.Deleted = filteredDeleted

	filteredAdded := make([]string, 0, len(changes.Added))
	for _, a := range changes.Added {
		if !renamedNew[a] {
			filteredAdded = append(filteredAdded, a)
		}
	}
	changes.Added = filteredAdded

	changes.Renamed = append(changes.Renamed, renameInfos...)
}

// storeBlobWithIndex helps store a blob in .evo/objects/blobs when partial commits are used,
// or when we want to ensure consistent hashing with the index. In partial commits, we still want
// the advantage of the index to avoid re-hashing if not changed.
func storeBlobWithIndex(repoPath, relPath string) (string, error) {
	idx, err := LoadIndex(repoPath)
	if err != nil {
		return "", err
	}
	hash, err := getOrComputeHash(repoPath, relPath, idx)
	if err != nil {
		return "", err
	}
	if err := SaveIndex(repoPath, idx); err != nil {
		return "", err
	}

	// ensure the blob is actually written to .evo/objects/blobs
	evoPath := filepath.Join(repoPath, EvoDir)
	data, err := os.ReadFile(filepath.Join(repoPath, relPath))
	if err != nil {
		return "", err
	}
	blobPath := filepath.Join(evoPath, "objects", "blobs", hash)
	if _, serr := os.Stat(blobPath); os.IsNotExist(serr) {
		if werr := os.WriteFile(blobPath, data, 0644); werr != nil {
			return "", werr
		}
	}
	return hash, nil
}

// GetWorkingChanges returns the current changes in the working directory compared to ACTIVE.
func GetWorkingChanges(repoPath string) (*FileChanges, error) {
	ignorePatterns, err := LoadIgnorePatterns(repoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load evo-ignore: %v", err)
	}

	evoPath := filepath.Join(repoPath, EvoDir)
	activeRef, _ := ReadRef(evoPath, ActiveRef)
	var baseTree *Tree
	if activeRef == "" {
		baseTree = &Tree{Files: make(map[string]string)}
	} else {
		c, err := ReadCommitObject(evoPath, activeRef)
		if err != nil {
			return nil, err
		}
		bt, err := ReadTreeObject(evoPath, c.TreeHash)
		if err != nil {
			return nil, err
		}
		baseTree = bt
	}

	// We'll do an index-based approach so we don't re-hash everything if unmodified
	idx, err := LoadIndex(repoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load index: %v", err)
	}

	// Walk working directory
	wdMap := make(map[string]string)
	err = filepath.Walk(repoPath, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if p == filepath.Join(repoPath, EvoDir) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(repoPath, p)
		if strings.HasPrefix(rel, EvoDir) {
			return nil
		}
		if MatchesIgnorePatterns(rel, ignorePatterns) {
			return nil
		}

		h, herr := getOrComputeHash(repoPath, rel, idx)
		if herr != nil {
			// skip or handle error
			return nil
		}
		wdMap[rel] = h
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Save index after we've scanned
	if serr := SaveIndex(repoPath, idx); serr != nil {
		return nil, serr
	}

	fc := &FileChanges{}
	// detect added/modified
	for wdPath, wdHash := range wdMap {
		baseHash, ok := baseTree.Files[wdPath]
		if !ok {
			fc.Added = append(fc.Added, wdPath)
		} else if baseHash != wdHash {
			fc.Modified = append(fc.Modified, wdPath)
		}
	}
	// detect deleted
	for basePath := range baseTree.Files {
		if _, ok := wdMap[basePath]; !ok {
			fc.Deleted = append(fc.Deleted, basePath)
		}
	}

	// rename detection: if a file in base was "deleted" but a file in wd was "added" with
	// the same hash, treat it as a rename.
	deletedFiles := make(map[string]string)
	for _, oldPath := range fc.Deleted {
		deletedFiles[baseTree.Files[oldPath]] = oldPath
	}

	addedFiles := make(map[string]string)
	for _, newPath := range fc.Added {
		addedFiles[wdMap[newPath]] = newPath
	}

	var renameInfos []RenameInfo
	for oldHash, oldPath := range deletedFiles {
		if newPath, ok := addedFiles[oldHash]; ok {
			renameInfos = append(renameInfos, RenameInfo{
				OldPath: oldPath,
				NewPath: newPath,
			})
		}
	}

	// filter out from fc.Deleted / fc.Added
	renamedOld := map[string]bool{}
	renamedNew := map[string]bool{}
	for _, ri := range renameInfos {
		renamedOld[ri.OldPath] = true
		renamedNew[ri.NewPath] = true
	}

	var finalDeleted []string
	for _, d := range fc.Deleted {
		if !renamedOld[d] {
			finalDeleted = append(finalDeleted, d)
		}
	}
	fc.Deleted = finalDeleted

	var finalAdded []string
	for _, a := range fc.Added {
		if !renamedNew[a] {
			finalAdded = append(finalAdded, a)
		}
	}
	fc.Added = finalAdded

	fc.Renamed = append(fc.Renamed, renameInfos...)

	return fc, nil
}

// fileBlobHash is a direct read+hash approach. It's used in getOrComputeHash if the modtime changed.
func fileBlobHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// For large file detection we also have handleLargeFiles() in largefiles.go, triggered in merges or commits if needed.

// GetStagedChanges returns the list of files recorded in .evo/staging/index
func GetStagedChanges(repoPath string) ([]string, error) {
	stPath := filepath.Join(repoPath, EvoDir, "staging", "index")
	if _, err := os.Stat(stPath); os.IsNotExist(err) {
		return nil, nil
	}
	data, err := os.ReadFile(stPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	var out []string
	for _, l := range lines {
		if l != "" {
			out = append(out, l)
		}
	}
	return out, nil
}
