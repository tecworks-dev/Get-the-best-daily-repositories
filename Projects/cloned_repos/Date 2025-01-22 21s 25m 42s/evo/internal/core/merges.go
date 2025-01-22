// Handle merges, ephemeral workspace merges, conflict detection, etc.
package core

import (
	"crypto/sha256"
	"encoding/hex"
	"evo/internal/plugins"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// MergeWorkspace merges the workspace commit with the current ACTIVE commit.
func MergeWorkspace(repoPath, workspaceName string) error {
	// Acquire a repository-wide lock using our new file-based lock approach.
	if err := lockRepo(repoPath); err != nil {
		return fmt.Errorf("cannot lock repository: %v", err)
	}
	defer unlockRepo(repoPath)

	evoPath := filepath.Join(repoPath, EvoDir)
	wsRef := "workspaces/" + workspaceName
	wsActive, _ := ReadRef(evoPath, wsRef)
	if wsActive == "" {
		return fmt.Errorf("workspace '%s' not found or empty", workspaceName)
	}

	currentActive, _ := ReadRef(evoPath, ActiveRef)
	if currentActive == wsActive {
		// nothing to merge
		return nil
	}

	// We'll do a 2-parent merge: (currentActive, wsActive)
	parents := []string{}
	if currentActive != "" {
		parents = append(parents, currentActive)
	}
	parents = append(parents, wsActive)

	mergeHash, err := doMerge(repoPath, parents, fmt.Sprintf("Merge workspace '%s'", workspaceName))
	if err != nil {
		return err
	}

	// Clear workspace ref
	_ = WriteRef(evoPath, wsRef, "")
	// Update ACTIVE to the result of the merge
	_ = WriteRef(evoPath, ActiveRef, mergeHash)
	return nil
}

// doMerge is an N-way merge. We sequentially merge the trees. For example, if parentHashes=[A,B,C],
// we start from A's tree, merge in B's, then merge in C's. A more sophisticated approach might do a true N-base merge.
func doMerge(repoPath string, parentHashes []string, message string) (string, error) {
	if len(parentHashes) == 0 {
		return "", fmt.Errorf("no parents to merge")
	}
	evoPath := filepath.Join(repoPath, EvoDir)

	// Start from the first parent's tree
	baseCommit, err := ReadCommitObject(evoPath, parentHashes[0])
	if err != nil {
		return "", err
	}
	baseTree, err := ReadTreeObject(evoPath, baseCommit.TreeHash)
	if err != nil {
		return "", err
	}
	mergedTree := CloneTree(baseTree)

	// Iteratively merge each subsequent parent's tree
	for i := 1; i < len(parentHashes); i++ {
		pc, err := ReadCommitObject(evoPath, parentHashes[i])
		if err != nil {
			return "", err
		}
		pt, err := ReadTreeObject(evoPath, pc.TreeHash)
		if err != nil {
			return "", err
		}
		merged, err := mergeTwoTrees(repoPath, mergedTree, pt)
		if err != nil {
			return "", err
		}
		mergedTree = merged
	}

	// Write final merged tree
	newTreeHash, err := writeTreeObject(evoPath, mergedTree)
	if err != nil {
		return "", err
	}

	// Create a new commit referencing all parents
	c := &Commit{
		Message:   message,
		Author:    "Evo Merge <merge@evo>", // generic placeholder
		Timestamp: time.Now().UTC(),
		Parents:   parentHashes,
		TreeHash:  newTreeHash,
	}

	// For hashing, we incorporate all commit metadata, including sorted parents
	raw := fmt.Sprintf("%s|%s|%v|%s", c.Message, c.Author, c.Timestamp.UnixNano(), c.TreeHash)
	sort.Strings(c.Parents)
	for _, p := range c.Parents {
		raw += "|" + p
	}
	sum := sha256.Sum256([]byte(raw))
	c.Hash = hex.EncodeToString(sum[:])

	if err := writeCommitObject(evoPath, c); err != nil {
		return "", err
	}

	return c.Hash, nil
}

// mergeTwoTrees merges t2 into t1, returning a new tree. If there's a conflict that cannot be
// structurally merged, we insert conflict markers into the working directory. We do not update
// the final blob in that case, leaving the user to resolve and commit later.
func mergeTwoTrees(repoPath string, t1, t2 *Tree) (*Tree, error) {
	evoPath := filepath.Join(repoPath, EvoDir)
	result := CloneTree(t1) // start with t1's version

	for f, bh2 := range t2.Files {
		bh1, ok := result.Files[f]
		if !ok {
			// new file in t2 not in t1
			result.Files[f] = bh2
			continue
		}

		// same hash => no conflict
		if bh1 == bh2 {
			continue
		}

		// We have a difference => attempt structural merges
		blob1Path := filepath.Join(evoPath, "objects", "blobs", bh1)
		blob2Path := filepath.Join(evoPath, "objects", "blobs", bh2)

		b1, err := os.ReadFile(blob1Path)
		if err != nil {
			return nil, fmt.Errorf("reading blob1 for %s: %w", f, err)
		}
		b2, err := os.ReadFile(blob2Path)
		if err != nil {
			return nil, fmt.Errorf("reading blob2 for %s: %w", f, err)
		}

		merged, mErr := structuralMerge(f, b1, b2)
		if mErr != nil {
			// structural merge not possible => produce conflict markers
			conflictPath := filepath.Join(repoPath, f)
			markers, cErr := createConflictMarkers(conflictPath, b1, b2)
			if cErr != nil {
				// fallback => prefer t2
				result.Files[f] = bh2
				continue
			}
			// Write conflict-markered file to working dir
			if err := os.WriteFile(conflictPath, []byte(markers), 0644); err != nil {
				// fallback => prefer t2
				result.Files[f] = bh2
				continue
			}
			// We do NOT update the tree with a new blob => user must fix, then commit
			if err := addConflictMarker(repoPath, f); err != nil {
				// If we can't add the marker for some reason, we just continue
			}
		} else {
			// structural merge succeeded => store new blob
			newHash, err := storeMergedBlob(filepath.Join(evoPath, "objects", "blobs"), merged)
			if err != nil {
				return nil, fmt.Errorf("storing merged blob for %s: %w", f, err)
			}
			result.Files[f] = newHash
		}
	}
	// if a file is in t1 but not in t2 => keep t1's version, no explicit removal here
	return result, nil
}

// createConflictMarkers produces typical conflict markers for a 2-sided conflict:
// <<<<<<< Ours
//
//	(ours content)
//
// =======
//
//	(theirs content)
//
// >>>>>>> Theirs
func createConflictMarkers(path string, ours, theirs []byte) (string, error) {
	oursLines := strings.Split(string(ours), "\n")
	theirsLines := strings.Split(string(theirs), "\n")

	var sb strings.Builder
	sb.WriteString("<<<<<<< Ours\n")
	sb.WriteString(strings.Join(oursLines, "\n"))
	sb.WriteString("\n=======\n")
	sb.WriteString(strings.Join(theirsLines, "\n"))
	sb.WriteString("\n>>>>>>> Theirs\n")

	return sb.String(), nil
}

// addConflictMarker creates an empty file in .evo/conflicts/<filename> so we can track conflicts in "evo conflicts list".
func addConflictMarker(repoPath, relPath string) error {
	conflictsDir := filepath.Join(repoPath, EvoDir, "conflicts")
	if err := os.MkdirAll(conflictsDir, 0755); err != nil {
		return err
	}
	conflictFile := filepath.Join(conflictsDir, relPath)
	return os.WriteFile(conflictFile, []byte("conflict"), 0644)
}

// structuralMerge is a placeholder that attempts advanced merges (e.g. JSON, YAML) or returns an error if it can't handle them.
func structuralMerge(filePath string, b1, b2 []byte) ([]byte, error) {
	// For advanced merges, we delegate to our plugin system first, then fallback.
	return plugins.MergeByExtension(filePath, b1, b2)
}

// storeMergedBlob writes the merged content to a new blob in .evo/objects/blobs
func storeMergedBlob(blobDir string, content []byte) (string, error) {
	sum := sha256.Sum256(content)
	hash := hex.EncodeToString(sum[:])
	out := filepath.Join(blobDir, hash)
	if _, err := os.Stat(out); os.IsNotExist(err) {
		if err := os.WriteFile(out, content, 0644); err != nil {
			return "", err
		}
	}
	return hash, nil
}
