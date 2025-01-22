package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// This function was removed or renamed. Reintroduce it:
func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	buf := make([]byte, 64*1024) // 64kb chunks
	for {
		n, err := f.Read(buf)
		if err != nil && err != io.EOF {
			return "", err
		}
		if n == 0 {
			break
		}
		h.Write(buf[:n])
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// handleLargeFiles checks the changed files, ...
func handleLargeFiles(repoPath string, changes *FileChanges) ([]string, error) {
    evoPath := filepath.Join(repoPath, EvoDir)
    var refs []string
    threshold := int64(5 * 1024 * 1024) // 5MB

    moveIfLarge := func(relPath string) error {
        p := filepath.Join(repoPath, relPath)
        fi, err := os.Stat(p)
        if err != nil {
            return nil // might have been deleted.
        }
        if fi.Size() < threshold {
            return nil
        }
        // It's large, store it in .evo/largefiles
        hashVal, err := hashFile(p) // now it's defined
        if err != nil {
            return err
        }
        dst := filepath.Join(evoPath, "largefiles", hashVal)
        if _, err := os.Stat(dst); os.IsNotExist(err) {
            // Move or copy the file
            if err := os.Rename(p, dst); err != nil {
                return err
            }
            // We can create a stub in the working directory referencing the LFS object
            stubContent := fmt.Sprintf("EVO-LFS:%s\n", hashVal)
            if err := os.WriteFile(p, []byte(stubContent), 0644); err != nil {
                return err
            }
        }
        refs = append(refs, hashVal)
        return nil
    }

    // For any newly-added or modified files, check if they’re too large:
    for _, f := range changes.Added {
        if err := moveIfLarge(f); err != nil {
            return refs, err
        }
    }
    for _, f := range changes.Modified {
        if err := moveIfLarge(f); err != nil {
            return refs, err
        }
    }
    // For deletes, do nothing
    return refs, nil
}
